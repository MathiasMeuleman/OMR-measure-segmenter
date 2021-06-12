import json
import logging
from music21 import converter
from pathlib import Path
from segmenter.dirs import root_dir
from pprint import pprint
from tqdm import tqdm


MUSIC_DATA_EXTENSIONS = ['.mxl', '.xml', '.musicxml']


class ScoreMapper:

    def __init__(self, directory, log_to_file=False):
        self.directory = directory
        self.score = None
        self.parts = None
        self.annotations = None
        self.groupings = None
        self.log_to_file = log_to_file
        self.logger = self.get_logger()

    def get_logger(self):
        logger = logging.getLogger('scoremapper')
        if self.log_to_file:
            file_handler = logging.FileHandler(Path(self.directory) / 'output.log', 'w')
            file_handler.setLevel('INFO')
            logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel('WARNING')
        logger.addHandler(stream_handler)
        logger.setLevel('DEBUG')
        return logger

    def parse_page_specification(self, specification):
        if '-' in specification:
            start, end = list(map(int, specification.split('-')))[0:2]
            if start is None or end is None or start > end:
                raise AssertionError('Invalid page specification: {}'.format(specification))
            return list(range(start, end + 1))
        return [int(specification)]

    def parse_groupings(self):
        """
        Parse groupings file. Assumes the annotations are already loaded. Groupings are separated per line and are:
        - A comma separated lists of ids of parts that need to be grouped together, optionally followed by a colon (:)
            indicating page specifications. If page specifications are not provided, the groupings hold for the entire
            score.
        - Page specifications are page numbers or ranges of page numbers, separated by commas. Page ranges are given as
            "[start]-[end]" with both start and end inclusive. A page number or range of page numbers can be negated
            prepending an exclamation mark (!), indicating the grouping does not hold for the given page or range. Note
            that when page specifications are provided, the grouping is not applied to the entire score anymore.
        Example:
            Flute 1,Flute 2             # Flute 1 and Flute 2 are grouped throughout the entire score
            Oboe 1,Oboe 2:10,11,13      # Oboe 1 and Oboe 2 are grouped at pages 10, 11 and 13
            Bassoon 1,Bassoon 2:3-9,!5  # Bassoon 1 and Bassoon 2 are grouped at page 3 through 9 inclusive, but not 5
        """
        groupings_path = Path(self.directory, 'groupings.txt')
        if groupings_path.is_file():
            with open(groupings_path) as file:
                lines = [line.rstrip() for line in file]
            groupings = [[] for _ in self.annotations]
            for i, line in enumerate(lines):
                grouping = line.split(':')[0].split(',')
                if ':' in line:
                    page_specifications = line.split(':')[1].split(',')
                    active_pages = [page for spec in page_specifications if not spec.startswith('!') for page in self.parse_page_specification(spec)]
                    exceptions = [page for spec in page_specifications if spec.startswith('!') for page in self.parse_page_specification(spec.split('!')[1])]
                    pages = list(set(active_pages) - set(exceptions))
                    for page in pages:
                        groupings[page - 1].append(grouping)
                else:
                    for j in range(len(self.annotations)):
                        groupings[j].append(grouping)
            self.groupings = groupings
        else:
            self.groupings = [[] for _ in self.annotations]

    def check_count_measures(self):
        """
        Pre-flight check to see if the amount of measures per part is as expected.
        """
        symbolic_measure_count = [len(part.recurse(classFilter='Measure')) for part in self.parts]
        annotation_measure_count = sum([system[1] for page in self.annotations for system in page])
        if annotation_measure_count != symbolic_measure_count[0] or any([symbolic_measure_count[i] != symbolic_measure_count[0] for i in range(len(symbolic_measure_count))]):
            self.logger.info('Found the following measure counts:\n\tAnnotated measures: {}\n\tSymbolic measures:'.format(annotation_measure_count))
            for i, part in enumerate(self.parts):
                self.logger.info('{}: {}'.format(part.id, symbolic_measure_count[i]))
            raise AssertionError('Could not match total number of measures')

    def get_score_path(self):
        music_path = next((f for f in Path(self.directory).iterdir() if f.suffix in MUSIC_DATA_EXTENSIONS), None)
        if not music_path.is_file():
            raise FileNotFoundError('Could not find supported music data file in directory ' + str(self.directory))
        return music_path

    def get_annotations_path(self):
        annotations_path = Path(self.directory, 'annotations.txt')
        if not annotations_path.is_file():
            raise FileNotFoundError('Could not find annotations file in directory ' + str(self.directory))
        return annotations_path

    def load_annotations(self):
        annotations_path = self.get_annotations_path()
        with open(annotations_path) as file:
            annotations = [list(map(lambda x: list(map(int, x.split(','))), line.rstrip().split(' '))) for line in file]
        self.annotations = annotations

    def load_score(self):
        music_path = self.get_score_path()
        self.logger.warning('Loading score found at: ' + str(music_path.resolve()))
        score = converter.parse(music_path)
        self.score = score
        parts = score.recurse(classFilter='Part')
        self.parts = list(parts)

        # celesta = parts.getElementById('P50-Staff2')
        # empty_measures = [len(parts.getElementById('P50-Staff2').measure(425).voices), len(parts.getElementById('P50-Staff1').measure(427).voices)]
        # pprint(vars(celesta.measure(425).flattenUnnecessaryVoices(force=True).notes[0].))

    def init(self):
        self.load_score()
        self.load_annotations()
        self.parse_groupings()
        self.check_count_measures()

    def list_parts(self):
        if self.score is None:
            raise AssertionError('Score not loaded')
        self.logger.info('\n'.join([str({'id': part.id, 'name': part.partName}) for part in self.parts]))


    @staticmethod
    def grouped_parts_is_empty(grouped_parts, start, end):
        """
        Determine if grouped parts (parts that are shown on a single staff) do not containing any non-rest musical
        information between start and end (indexing type values). If a measure contains voices, these are forcefully
        flattened, since notes in measures are only counted when they are top-level. Note visibility is checked through
        the `hideObjectOnPrint` option in the NoteStyle object, if set to `True`, the note is ignored.
        """
        return not any(len([note
                            for note in (m.flattenUnnecessaryVoices(force=True) if len(m.voices) > 0 else m).notes
                            if note._style is None or note._style.hideObjectOnPrint is not True]) > 0
                       for part in grouped_parts
                       for m in part.measures(start, end, collect=[], gatherSpanners=False, indicesNotNumbers=True))

    @staticmethod
    def generate_annotated_part_group(part_group, measure_count):
        return {'parts': [{'id': part.id, 'name': part.partName} for part in part_group], 'measures': measure_count}

    def get_part_groups(self, page):
        part_groups = []
        for grouping in self.groupings[page]:
            group_parts = [part for part in self.parts if part.id in grouping]
            part_groups.append(group_parts)
        grouped_ids = [id for ids in self.groupings[page] for id in ids]
        single_parts = [[part] for part in self.parts if part.id not in grouped_ids]
        part_groups.extend(single_parts)
        return part_groups

    def match_system(self, page_idx, system_idx, start):
        page_nr, system_nr = page_idx + 1, system_idx + 1
        part_groups = self.get_part_groups(page_idx)
        (part_count, measure_count) = self.annotations[page_idx][system_idx]
        empty_part_groups, non_empty_part_groups = [], []
        for group in part_groups:
            (empty_part_groups
             if self.grouped_parts_is_empty(group, start, start + measure_count)
             else non_empty_part_groups
             ).append(group)
        if len(non_empty_part_groups) > part_count:
            self.logger.warning([[part.id for part in part_group] for part_group in non_empty_part_groups])
            raise AssertionError('Found {} non-empty voices in system expected {} at most, at page: {}, system: {}, current_measure: {}'
                                 .format(len(non_empty_part_groups), part_count, page_nr, system_nr, start))
        if len(non_empty_part_groups) < part_count:
            self.logger.info('Found {} empty staffs at page {}, system {}'.format(part_count - len(non_empty_part_groups), page_nr, system_nr))
        matches = [self.generate_annotated_part_group(grouped_part, measure_count) for grouped_part in non_empty_part_groups]
        matches.extend([{'parts': None, 'measures': measure_count} for _ in range(part_count - len(non_empty_part_groups))])
        return matches

    def match_page(self, page_number):
        page_idx = page_number - 1
        current_measure = sum([system[1] for page in self.annotations[0:page_idx] for system in page])
        for i in range(len(self.annotations[page_idx])):
            start = current_measure + sum([system[1] for system in self.annotations[page_idx][0:i]])
            matched_parts = self.match_system(page_idx, i, start)
            pprint(matched_parts)

    def match_score(self):
        """
        Attempt to find a plausible mapping from the symbolic score to the annotations on measure level. In short this
        is done as follows:
         - Group all the parts (voices) based on the groupings listing that is provided.
         - For each page, for each system, find all grouped parts that any more musical content than just rests. These
            grouped parts are assumed to always appear on the page (since they have musical content, they logically
            cannot be left out)
         - If the annotations indicate that there are more grouped parts present in the system than we have found, there
            are empty grouped parts included as well (generally this is an editorial choice). For these we cannot
            determine which of the empty grouped parts are shown and which are not.
        """
        if self.score is None:
            raise AssertionError('Score not loaded')

        current_measure = 0
        pages = []
        for i, page in enumerate(tqdm(self.annotations)):
            page_number = i + 1
            systems = []
            for j, (voices, measures) in enumerate(page):
                system_number = j + 1
                matched_parts = self.match_system(i, j, current_measure)
                systems.append({'systemNumber': system_number, 'parts': matched_parts})
                current_measure += measures
            pages.append({'pageNumber': page_number, 'systems': systems})

        result = json.dumps({'pages': pages}, indent=2, sort_keys=True)
        with open(Path(self.directory) / 'score_mapping.json', 'w') as file:
            file.write(result)
        return result

    def get_matched_score(self):
        annotations_path = self.get_annotations_path()
        score_path = self.get_score_path()
        groupings_path = Path(self.directory, 'groupings.txt')
        mapping_path = Path(self.directory, 'score_mapping.json')
        needs_match = True
        if mapping_path.is_file():
            mapping_mtime = mapping_path.stat().st_mtime
            if annotations_path.stat().st_mtime < mapping_mtime and score_path.stat().st_mtime < mapping_mtime:
                needs_match = False
            if groupings_path.is_file() and groupings_path.stat().st_mtime < mapping_mtime:
                needs_match = False
        if needs_match:
            self.init()
            return self.match_score()
        else:
            self.logger.warning('Using existing mapping found at: {}'.format(mapping_path.resolve()))
            with open(mapping_path) as file:
                mapping = json.load(file)
            return mapping


def match_score(path):
    source = ScoreMapper(path, log_to_file=True)
    # source.init()
    # source.list_parts()
    source.get_matched_score()


def match_page(path, page):
    source = ScoreMapper(path)
    source.init()
    source.match_page(page)


if __name__ == '__main__':
    part_directories = [
        'bach_brandenburg_concerto_5_part_1',
        'holst_the_planets',
        'mozart_symphony_41',
        'tchaikovsky_ouverture_1812/edition_1',
        'temp'
    ]

    musicdata_directory = Path(root_dir).parent / 'OMR-measure-segmenter-data/musicdata'
    path = musicdata_directory / part_directories[0]
    match_score(path)
    # match_page(path, 41)

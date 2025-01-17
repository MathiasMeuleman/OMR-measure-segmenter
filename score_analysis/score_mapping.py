import json
import logging
from music21 import converter
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

from util.dirs import musicdata_dir

MUSIC_DATA_EXTENSIONS = ['.mxl', '.xml', '.musicxml']

none_part = "NONEPART"


class ScoreMappingPart:

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __str__(self):
        return 'Id: {}\nName: {}'.format(self.id, self.name)

    @staticmethod
    def from_json(json_data):
        return ScoreMappingPart(json_data['id'], json_data['name'])


class ScoreMappingStaff:

    def __init__(self, staffnumber, parts):
        self.staffnumber = staffnumber
        self.parts = parts

    def __str__(self):
        string = 'Staff: {}\nParts:'.format(self.staffnumber)
        for part in self.parts:
            string += '\n'
            if part is none_part:
                string += part
            else:
                string += part.__str__()
        return string

    @staticmethod
    def from_json(json_data):
        staffnumber = json_data['staffNumber']
        if json_data['parts'] is None:
            return ScoreMappingStaff(staffnumber, [])
        return ScoreMappingStaff(staffnumber, [ScoreMappingPart.from_json(part) for part in json_data['parts']])


class ScoreMappingSystem:

    def __init__(self, systemnumber, measurestart, measureend, staffs):
        self.systemnumber = systemnumber
        self.measurestart = measurestart
        self.measureend = measureend
        self.staffs = staffs

    def __str__(self):
        string = 'System: {}\nMeasures: {}-{}\nStaffs:'.format(self.systemnumber, self.measurestart, self.measureend)
        for staff in self.staffs:
            string += '\n' + staff.__str__()
        return string

    @staticmethod
    def from_json(json_data):
        staffs = [ScoreMappingStaff.from_json(staff) for staff in json_data['staffs']]
        return ScoreMappingSystem(json_data['systemNumber'], json_data['measureStart'], json_data['measureEnd'], staffs)


class ScoreMappingPage:

    def __init__(self, pagenumber, systems):
        self.pagenumber = pagenumber
        self.systems = systems

    def __str__(self):
        string = 'Page {}\nSystems:'.format(self.pagenumber)
        for s in self.systems:
            string += '\n' + s.__str__()
        return string

    @staticmethod
    def from_json(json_data):
        systems = [ScoreMappingSystem.from_json(system) for system in json_data['systems']]
        return ScoreMappingPage(json_data['pageNumber'], systems)


class ScoreMapper:

    def __init__(self, directory, log_to_file=False):
        self.initialized = False
        self.directory = Path(directory)
        self.score = None
        self.parts = None
        self.annotations = None
        self.groupings = None
        self.visible_parts_hint = None
        self.log_to_file = log_to_file
        self.logger = self.get_logger()

    def get_logger(self):
        logger = logging.getLogger('scoremapper')
        if self.log_to_file:
            file_handler = logging.FileHandler(self.directory / 'output.log', 'w')
            file_handler.setLevel('INFO')
            logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel('WARNING')
        logger.addHandler(stream_handler)
        logger.setLevel('DEBUG')
        return logger

    def parse_page_specification(self, specification):
        if '.' in specification:
            return [specification]
        if '-' in specification:
            start, end = list(map(int, specification.split('-')))[0:2]
            if start is None or end is None or start > end:
                raise AssertionError('Invalid page specification: {}'.format(specification))
            return ['{}.{}'.format(page, system + 1) for page in list(range(start, end + 1)) for system in range(len(self.annotations[page - 1]))]
        page = int(specification)
        return ['{}.{}'.format(page, system + 1) for system in range(len(self.annotations[page - 1]))]

    def parse_groupings(self):
        """
        Parse groupings file. Assumes the annotations are already loaded. Groupings are separated per line and are:
        - A comma separated lists of ids of parts that need to be grouped together, optionally followed by a colon (:)
            indicating page specifications. If page specifications are not provided, the groupings hold for the entire
            score. If ids of parts contain commas (,), the semicolon (;) can be used instead as separator.
        - Page specifications are page numbers or ranges of page numbers, separated by commas. Page ranges are given as
            "[start]-[end]" with both start and end inclusive. A page number or range of page numbers can be negated
            prepending an exclamation mark (!), indicating the grouping does not hold for the given page or range. Note
            that when page specifications are provided, the grouping is not applied to the entire score anymore.
        - Page specifications can be extended to include or exclude specific systems on a page as well. These are
            specified by a seperating dot (.).
        Example:
            Flute 1,Flute 2             # Flute 1 and Flute 2 are grouped throughout the entire score
            Oboe 1,Oboe 2:10,11,13      # Oboe 1 and Oboe 2 are grouped at pages 10, 11 and 13
            Bassoon 1,Bassoon 2:3-9,!5  # Bassoon 1 and Bassoon 2 are grouped at page 3 through 9 inclusive, but not 5
            Violoncello,Basso:1-5,!2,!2.1
        """
        groupings_path = self.directory / 'groupings.txt'
        if groupings_path.is_file():
            with open(groupings_path) as file:
                lines = [line.rstrip() for line in file]
            groupings = [[[] for _ in page] for page in self.annotations]
            for i, line in enumerate(lines):
                grouping_str = line.split(':')[0]
                separator = ';' if ';' in grouping_str else ','
                grouping = grouping_str.split(separator)
                if ':' in line:
                    page_specifications = line.split(':')[1].split(',')
                    active_specs = [page for spec in page_specifications if not spec.startswith('!') for page in self.parse_page_specification(spec)]
                    exceptions = [page for spec in page_specifications if spec.startswith('!') for page in self.parse_page_specification(spec.split('!')[1])]
                    specs = list(set(active_specs) - set(exceptions))
                    for spec in specs:
                        page, system = list(map(int, spec.split('.')))
                        groupings[page - 1][system - 1].append(grouping)
                else:
                    for j in range(len(self.annotations)):
                        for k in range(len(self.annotations[j])):
                            groupings[j][k].append(grouping)
            self.groupings = groupings
        else:
            self.groupings = [[[] for _ in self.annotations[i]] for i, _ in enumerate(self.annotations)]

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
        music_path = next((f for f in self.directory.iterdir() if f.suffix in MUSIC_DATA_EXTENSIONS), None)
        if music_path is None or not music_path.is_file():
            music_path = self.directory / 'stage2s'
            if not music_path.is_dir():
                raise FileNotFoundError('Could not find supported music data in directory ' + str(self.directory))
        return music_path

    def get_annotations_path(self):
        annotations_path = self.directory / 'annotations.txt'
        if not annotations_path.is_file():
            raise FileNotFoundError('Could not find annotations file in directory ' + str(self.directory))
        return annotations_path

    def load_annotations(self):
        annotations_path = self.get_annotations_path()
        with open(annotations_path) as file:
            annotations = [list(map(lambda x: list(map(int, x.split(','))), line.rstrip().split(' '))) for line in file]
        self.annotations = annotations

    def load_visibile_parts_hint(self):
        """
        Format for visible parts hint is either [start]-[end]:id1,id2,id3 or [page].[system]:id1,id2,id3
        """
        visible_parts_hint = [[[] for _ in page] for page in self.annotations]
        visibible_parts_hint_path = self.directory / 'visible_parts_hint.txt'
        if visibible_parts_hint_path.is_file():
            with open(visibible_parts_hint_path) as file:
                hints = file.read().splitlines()
            for hint in hints:
                page_str, part_str = hint.split(':')[0:2]
                ids = [s.replace(';', ',') for s in part_str.split(',')]
                if '-' in page_str:
                    start, end = [int(x) for x in page_str.split('-')[0:2]]
                    for page in range(start, end + 1):
                        for system in range(len(visible_parts_hint[page - 1])):
                            visible_parts_hint[page - 1][system] = ids
                else:
                    page, system = [int(x) for x in hint.split(':')[0].split('.')[0:2]]
                    visible_parts_hint[page - 1][system - 1] = ids
        self.visible_parts_hint = visible_parts_hint

    def load_score(self):
        music_path = self.get_score_path()
        self.logger.warning('Loading score found at: ' + str(music_path.resolve()))
        score = converter.parse(music_path)
        self.score = score
        parts = score.recurse(classFilter='Part')
        self.parts = list(parts)

    def init(self):
        if self.initialized:
            return
        self.load_score()
        self.load_annotations()
        self.parse_groupings()
        self.load_visibile_parts_hint()
        self.check_count_measures()
        self.initialized = True

    def list_parts(self, ids_only=False):
        if self.score is None:
            raise AssertionError('Score not loaded')
        if ids_only:
            log = '\n'.join([part.id for part in self.parts])
        else:
            log = '\n'.join([str({'id': part.id, 'name': part.partName}) for part in self.parts])
        self.logger.info(log)

    @staticmethod
    def get_part_measures(part, start, end):
        return part.measures(start, end, collect=[], gatherSpanners=False, indicesNotNumbers=True)

    @staticmethod
    def measure_is_empty(measure):
        return len([note
                    for note in (measure.flattenUnnecessaryVoices(force=True) if len(measure.voices) > 0 else measure).notes
                    if note._style is None or note._style.hideObjectOnPrint is not True]) <= 0

    @staticmethod
    def grouped_parts_is_empty(grouped_parts, start, end):
        """
        Determine if grouped parts (parts that are shown on a single staff) do not containing any non-rest musical
        information between start and end (indexing type values). If a measure contains voices, these are forcefully
        flattened, since notes in measures are only counted when they are top-level. Note visibility is checked through
        the `hideObjectOnPrint` option in the NoteStyle object, if set to `True`, the note is ignored.
        """
        return all(ScoreMapper.measure_is_empty(measure)
                       for part in grouped_parts
                       for measure in ScoreMapper.get_part_measures(part, start, end))

    @staticmethod
    def generate_annotated_part_group(part_group, staff):
        return {'parts': [{'id': part.id, 'name': part.partName} for part in part_group], 'staffNumber': staff}

    def get_part_groups(self, page, system):
        part_groups = []
        for grouping in self.groupings[page][system]:
            group_parts = [part for part in self.parts if part.id in grouping]
            part_groups.append(group_parts)
        grouped_ids = [id for ids in self.groupings[page][system] for id in ids]
        single_parts = [[part] for part in self.parts if part.id not in grouped_ids]
        part_groups.extend(single_parts)
        return self.sort_part_groups(part_groups)

    def sort_part_key(self, order):
        def _sort_part_key(group):
            return min([order.index(p.id) for p in group])
        return _sort_part_key

    def sort_part_groups(self, part_groups):
        with open(self.directory / 'part_order.txt') as f:
            order = f.read().splitlines()
        return sorted(part_groups, key=self.sort_part_key(order))

    def match_system(self, page_idx, system_idx, start):
        page_nr, system_nr = page_idx + 1, system_idx + 1
        part_groups = self.get_part_groups(page_idx, system_idx)
        (part_count, measure_count) = self.annotations[page_idx][system_idx]

        empty_part_groups = [g for g in part_groups if self.grouped_parts_is_empty(g, start, start + measure_count)]
        part_groups_visible = [not self.grouped_parts_is_empty(g, start, start + measure_count) for g in part_groups]
        missing_visible_count = part_count - sum(part_groups_visible)
        if missing_visible_count > 0:
            self.logger.info('Found {} empty staffs at page {}, system {}'.format(missing_visible_count, page_nr, system_nr))
        for id in self.visible_parts_hint[page_idx][system_idx]:
            idx = [parts[0].id for parts in part_groups].index(id)
            part_groups_visible[idx] = True
        missing_visible_count = part_count - sum(part_groups_visible)
        if len(empty_part_groups) == missing_visible_count:
            part_groups_visible = [True] * part_count
            missing_visible_count = 0
        if missing_visible_count > 0:
            self.logger.warning('Could not match staffs to part groups: Page {}, System {}'.format(page_nr, system_nr))
        while missing_visible_count > 0:
            idx = part_groups_visible.index(False)
            part_groups_visible[idx] = True
            missing_visible_count -= 1

        matches = []
        staff = 1
        for (visible, group) in zip(part_groups_visible, part_groups):
            if visible:
                matches.append(self.generate_annotated_part_group(group, staff))
                staff += 1

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
                systems.append({'systemNumber': system_number, 'staffs': matched_parts, 'measureStart': current_measure, 'measureEnd': current_measure + measures})
                current_measure += measures
            pages.append({'pageNumber': page_number, 'systems': systems})

        result = json.dumps({'pages': pages}, indent=2, sort_keys=True)
        with open(self.directory / 'score_mapping.json', 'w') as file:
            file.write(result)
        return result

    def get_matched_score(self):
        annotations_path = self.get_annotations_path()
        score_path = self.get_score_path()
        groupings_path = self.directory / 'groupings.txt'
        mapping_path = self.directory / 'score_mapping.json'
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

    def get_measure_count(self):
        self.init()
        mapping = self.get_matched_score()
        symbolic_measures = sum([len(part.recurse(classFilter='Measure')) for part in self.parts])
        measures_count = sum([
            (int(system['measureEnd']) - int(system['measureStart'])) * len(system['staffs'])
            for page in mapping['pages']
            for system in page['systems']
        ])
        part_map = {part.id: part for part in self.parts}
        empty_measures = 0
        for page in mapping['pages']:
            for system in page['systems']:
                start, end = int(system['measureStart']), int(system['measureEnd'])
                for staff in system['staffs']:
                    if staff['parts'] is None:
                        empty_measures += (end - start)
                    else:
                        parts_group = [part_map[part['id']] for part in staff['parts']]
                        empty_measures += sum([self.grouped_parts_is_empty(parts_group, start, start + 1) for start in range(start, end)])
        return {
            'symbolic': symbolic_measures,
            'written': measures_count,
            'empty': empty_measures
        }


def match_page(path, page):
    source = ScoreMapper(path)
    source.init()
    source.match_page(page)


def count_measures(path):
    source = ScoreMapper(path)
    print(source.get_measure_count())


if __name__ == '__main__':
    part_directories = [
        'bach_brandenburg_concerto_5_part_1',
        'beethoven_symphony_1/part_1',
        'beethoven_symphony_1/part_2',
        'beethoven_symphony_1/part_3',
        'beethoven_symphony_1/part_4',
        'beethoven_symphony_2/part_1',
        'beethoven_symphony_2/part_2',
        'beethoven_symphony_2/part_3',
        'beethoven_symphony_2/part_4',
        'beethoven_symphony_3/part_1',
        'beethoven_symphony_3/part_2',
        'beethoven_symphony_3/part_3',
        'beethoven_symphony_3/part_4',
        'beethoven_symphony_4/part_1',
        'beethoven_symphony_4/part_2',
        'beethoven_symphony_5/part_1',
        'beethoven_symphony_5/part_2',
        'beethoven_symphony_5/part_3',
        'beethoven_symphony_5/part_4',
        'beethoven_symphony_6/part_1',
        'beethoven_symphony_6/part_2',
        'beethoven_symphony_6/part_3',
        'beethoven_symphony_6/part_4',
        'beethoven_symphony_6/part_5',
        'beethoven_symphony_7/part_1',
        'beethoven_symphony_7/part_2',
        'beethoven_symphony_7/part_3',
        'beethoven_symphony_7/part_4',
        'beethoven_symphony_8/part_1',
        'beethoven_symphony_8/part_2',
        'beethoven_symphony_8/part_3',
        'beethoven_symphony_8/part_4',
        'beethoven_symphony_9/part_1',
        'beethoven_symphony_9/part_2',
        'beethoven_symphony_9/part_3',
        'beethoven_symphony_9/part_4',
        'brahms_symphony_3',
        'bruckner_symphony_5',
        'bruckner_symphony_9',
        'holst_the_planets',
        'mahler_symphony_4',
        'mozart_symphony_41',
        'tchaikovsky_ouverture_1812',
    ]

    for part in part_directories:
        path = musicdata_dir / part
        source = ScoreMapper(path, log_to_file=True)
        print(part)
        print(source.get_measure_count())

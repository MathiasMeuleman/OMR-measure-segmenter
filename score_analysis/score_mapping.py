import json
from music21 import converter
from pathlib import Path
from segmenter.dirs import root_dir


MUSIC_DATA_EXTENSIONS = ['.mxl', '.xml', '.musicxml']


class ScoreMapper:

    def __init__(self, directory):
        self.directory = directory
        self.score = None
        self.annotations = None
        self.groupings = None

    def parse_groupings(self):
        """
        Parse groupings file. Assumes the annotations are already loaded
        """
        groupings_path = Path(self.directory, 'groupings.txt')
        if groupings_path.is_file():
            with open(groupings_path) as file:
                lines = [line.rstrip() for line in file]
            default_grouping = [line.split(':')[0].split(',') for line in lines]
            exception_map = [[] for _ in self.annotations]
            for i, line in enumerate([line for line in lines if ':' in line]):
                pages = list(map(int, line.split(':')[1].split(',')))
                for page in pages:
                    exception_map[page - 1].append(i)
            groupings = [default_grouping.copy() for _ in self.annotations]
            for i, groups in enumerate(exception_map):
                for group in groups:
                    groupings[i][group] = []
            self.groupings = [[group for group in grouping if group] for grouping in groupings]

    def check_count_measures(self):
        """
        Pre-flight check to see if the amount of measures per part is as expected.
        """
        parts = self.score.recurse(classFilter='Parts')
        symbolic_measure_count = [len(part.recurse(classFilter='Measure')) for part in parts]
        annotation_measure_count = sum([system[1] for page in self.annotations for system in page])
        if any([symbolic_measure_count[i] != symbolic_measure_count[0] for i in range(len(symbolic_measure_count))]):
            print('Found the following measure counts:\n\tAnnotated measures: {}\n\tSymbolic measures:'.format(annotation_measure_count))
            for i, part in enumerate(parts):
                print(part.id, symbolic_measure_count[i])
            raise AssertionError('')

    def load_score(self):
        music_path = next((f for f in Path(self.directory).iterdir() if f.suffix in MUSIC_DATA_EXTENSIONS), None)
        annotations_path = Path(self.directory, 'annotations.txt')
        if not music_path.is_file():
            raise FileNotFoundError('Could not find supported music data file in directory ' + str(self.directory))
        if not annotations_path.is_file():
            raise FileNotFoundError('Could not find annotations file in directory ' + str(self.directory))

        score = converter.parse(music_path)
        self.score = score

        with open(annotations_path) as file:
            annotations = [list(map(lambda x: list(map(int, x.split(','))), line.rstrip().split(' '))) for line in file]
        self.annotations = annotations
        self.parse_groupings()
        self.check_count_measures()

    @staticmethod
    def grouped_parts_is_empty(grouped_parts, start, end):
        """
        Determine if grouped parts (parts that are shown on a single staff) do not containing any non-rest musical
        information between start and end (indexing type values). If a measure contains voices, these are forcefully
        flattened, since notes in measures are only counted when they are top-level.
        """
        return not any(len((m.flattenUnnecessaryVoices(force=True) if len(m.voices) > 0 else m).notes) > 0
                       for part in grouped_parts
                       for m in part.measures(start, end, collect=[], gatherSpanners=False, indicesNotNumbers=True))

    @staticmethod
    def generate_annotated_part_group(part_group, measure_count):
        return {'parts': [{'id': part.id, 'name': part.partName} for part in part_group], 'measures': measure_count}

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
        :return:
        """
        parts = list(self.score.recurse(classFilter='Part'))

        current_measure = 0
        pages = []
        for i, page in enumerate(self.annotations):
            page_number = i + 1

            part_groups = []
            for grouping in self.groupings[i]:
                group_parts = [part for part in parts if part.id in grouping]
                part_groups.append(group_parts)
            grouped_ids = [id for ids in self.groupings[i] for id in ids]
            single_parts = [[part] for part in parts if part.id not in grouped_ids]
            part_groups.extend(single_parts)

            systems = []
            for j, (voices, measures) in enumerate(page):
                system_number = j + 1
                empty_part_groups, non_empty_part_groups = [], []
                for group in part_groups:
                    (empty_part_groups
                        if self.grouped_parts_is_empty(group, current_measure, current_measure + measures)
                        else non_empty_part_groups
                     ).append(group)
                if len(non_empty_part_groups) > voices:
                    print([part.id for part_group in non_empty_part_groups for part in part_group])
                    raise AssertionError('Found {} non-empty voices in system expected {} at most, at page: {}, system: {}, current_measure: {}'
                                         .format(len(non_empty_part_groups), voices, page_number, system_number, current_measure))
                if len(non_empty_part_groups) < voices:
                    print('Found {} empty staffs at page {}, system {}'.format(voices - len(non_empty_part_groups), page_number, system_number))
                matched_parts = [self.generate_annotated_part_group(grouped_part, measures) for grouped_part in non_empty_part_groups]
                matched_parts.extend([{'parts': None, 'measures': measures} for _ in range(voices - len(non_empty_part_groups))])
                systems.append({'systemNumber': system_number, 'parts': matched_parts})
                current_measure += measures
            pages.append({'pageNumber': page_number, 'systems': systems})

        result = json.dumps({'pages': pages}, indent=2, sort_keys=True)
        with open(Path(self.directory) / 'score_mapping.json', 'w') as file:
            file.write(result)


if __name__ == '__main__':
    path = Path(Path(root_dir).parent, 'OMR-measure-segmenter-data/musicdata/tchaikovsky_ouverture_1812/edition_1').resolve()
    source = ScoreMapper(path)
    source.load_score()
    source.match_score()

import json
from itertools import groupby
from pathlib import Path

from PIL import Image
from operator import attrgetter
from tqdm import tqdm

from score_analysis.barline_detector import System
from score_analysis.measure_detector import Measure
from score_analysis.score_image import ScoreImage
from score_analysis.staff_detector import Staff
from util.dirs import data_dir, get_musicdata_scores, page_sort_key


class MeasureExtractor:

    def __init__(self, image, measure_path, output_path):
        self.score_image = image if isinstance(image, ScoreImage) else ScoreImage(image)
        self.image = self.score_image.image
        self.measure_path = Path(measure_path)
        self.output_path = Path(output_path)
        self.measures = None
        self.systems = None

    class System:

        def __init__(self, measures):
            self.measures = measures
            self.top = measures[0].top
            self.bottom = measures[-1].bottom

    def load_measures(self):
        if not self.measure_path.exists():
            raise FileNotFoundError('Could not find measures at: {}'.format(self.measure_path))
        with open(self.measure_path) as f:
            self.measures = [Measure.from_json(json_data) for json_data in json.load(f)['measures']]
        self.measures.sort(key=attrgetter('system', 'start', 'top'))
        self.systems = [self.System(list(v)) for k, v in groupby(self.measures, key=attrgetter('system'))]

        for i, system in enumerate(self.systems):
            bar = 0
            staff = 0
            prev_top = system.measures[0].top + 1
            for j, measure in enumerate(system.measures):
                if measure.top < prev_top:
                    bar += 1
                    staff = 1
                else:
                    staff += 1
                prev_top = measure.top
                measure.bar = bar
                measure.staff = staff

    def get_measures_bb(self):
        """Dumb bounding box finder, maximizes spacing around measures equally, without overlap."""
        top_measures = []
        bottom_measures = []
        for i, system in enumerate(self.systems):
            prev_system = None if i == 0 else self.systems[i - 1]
            next_system = None if i == len(self.systems) - 1 else self.systems[i + 1]
            for j, measure in enumerate(system.measures):
                prev_measure = None if measure.top == system.top else system.measures[j - 1]
                next_measure = None if measure.bottom == system.bottom else system.measures[j + 1]
                if prev_measure is not None:
                    bb_top = measure.top - (measure.top - prev_measure.bottom) // 2
                elif prev_system is not None:
                    bb_top = measure.top - (measure.top - prev_system.bottom) // 2
                else:
                    # Edge case for top measures of first system handled separately
                    bb_top = measure.top
                    top_measures.append(measure)

                if next_measure is not None:
                    bb_bottom = measure.bottom + (next_measure.top - measure.bottom) // 2
                elif next_system is not None:
                    bb_bottom = measure.bottom + (next_system.top - measure.bottom) // 2
                else:
                    # Edge case for bottom measures of last system handled separately
                    bb_bottom = measure.bottom
                    bottom_measures.append(measure)
                measure.bb = (measure.start, bb_top, measure.end, bb_bottom)
        max_diff = max([max(m.top - m.bb[1], m.bb[3] - m.bottom) for s in self.systems for m in s.measures])
        for measure in top_measures:
            new_top = max(0, measure.top - max_diff)
            measure.bb = (measure.bb[0], new_top) + measure.bb[2:]
        for measure in bottom_measures:
            new_bottom = min(self.image.height - 1, measure.bottom + max_diff)
            measure.bb = measure.bb[0:3] + tuple([new_bottom])

    def extract_measures(self):
        if self.measures is None:
            self.load_measures()
        self.get_measures_bb()
        for system in self.systems:
            for i, measure in enumerate(system.measures):
                name = 'system_{}_measure_{}.png'.format(measure.system + 1, i + 1)
                measure_image = self.image.crop(measure.bb)
                measure_image.save(self.output_path / name)


def filter_pages(part):
    """Filter out pages from this part that do not adhere to the true annotations."""
    with open(part / 'annotations.txt') as f:
        annotations = [list(map(lambda x: list(map(int, x.split(','))), line.rstrip().split(' '))) for line in f]
    filtered_pages = []
    for i, page in enumerate(sorted((part / 'pages').iterdir(), key=page_sort_key)):
        json_name = page.stem + '.json'
        with open(part / 'staffs' / json_name) as f:
            staffs = [Staff.from_json(json_data) for json_data in json.load(f)['staffs']]
        with open(part / 'barlines' / json_name) as f:
            systems = [System.from_json(json_data) for json_data in json.load(f)['systems']]
        page_annotations = annotations[i]
        include = True
        if len(page_annotations) != len(systems):
            include = False
        else:
            if not all([len(sys.barlines) - 1 == ann[1] for sys, ann in zip(systems, page_annotations)]):
                include = False
            if len(staffs) != sum([ann[0] for ann in page_annotations]):
                include = False

        if include:
            filtered_pages.append(page)
    return filtered_pages


if __name__ == '__main__':
    part = data_dir / 'sample'
    measures_path = part / 'measures'
    for page in tqdm(filter_pages(part)):
        measure_images_path = part / 'measure_images' / page.stem
        measure_images_path.mkdir(parents=True, exist_ok=True)
        measure_path = measures_path / (page.stem + '.json')
        extractor = MeasureExtractor(Image.open(page), measure_path, measure_images_path)
        extractor.extract_measures()


import json
from itertools import groupby

from PIL import Image
from operator import attrgetter

from music21 import converter
import numpy as np

from score_analysis.measure_clusterer import MeasureImage
from score_analysis.measure_detector import Measure
from score_analysis.score_mapping import ScoreMappingPage
from util.dirs import musicdata_dir, get_parts, get_score_dirs


class ClusterMeasure(MeasureImage):

    def __init__(self, measure, image, page, system, bar, staff, part, score_measures):
        super(ClusterMeasure, self).__init__(measure, image, page, system, bar, staff)
        self.idx = -1
        self.label = -1
        self.part = part
        self.score_measures = score_measures

    def __str__(self):
        return 'Part: {}\tPage: {}\tSystem: {}\tBar: {}\tStaff: {}'.format(self.part, self.page, self.system, self.bar, self.staff)


class ClusterEvaluator:

    def __init__(self, score_dirs):
        self.measures_path = score_dirs['measures']
        self.measure_images_path = score_dirs['measure_images']
        self.labels_path = score_dirs['labels']
        self.medoids_path = score_dirs['medoids']
        self.score_path = score_dirs['score_path']
        self.score_mapping_path = score_dirs['score_mapping']
        self.part_order_path = score_dirs['part_order']
        self.labels = None
        self.medoids = None
        self.measures = None
        self.score_mapping = None
        self.score = None
        self.parts = None

    def load_measures(self):
        print('Loading measures...')
        self.measures = []
        page_offset = 0
        for i, (measures_path, images_path, parts, score_mapping) in enumerate(zip(self.measures_path, self.measure_images_path, self.parts, self.score_mapping)):
            if not measures_path.exists():
                raise FileNotFoundError('No measures folder found at: {}'.format(self.measures_path))
            if not images_path.exists():
                raise FileNotFoundError('No images folder found at: {}'.format(self.measure_images_path))
            part_nr = i + 1
            for images_path in sorted(images_path.iterdir()):
                page_path = measures_path / (images_path.stem + '.json')
                if not images_path.exists():
                    raise FileNotFoundError('No image found at: {}'.format(images_path))
                page_nr = int(page_path.stem.split('_')[1])
                with open(page_path) as f:
                    measures = [Measure.from_json(json_data) for json_data in json.load(f)['measures']]
                measures.sort(key=attrgetter('system', 'start', 'top'))
                systems = [list(v) for k, v in groupby(measures, key=attrgetter('system'))]
                for j, system in enumerate(systems):
                    system_nr = j + 1
                    bar = 0
                    staff = 0
                    prev_top = system[0].top + 1
                    for k, measure in enumerate(system):
                        if measure.top < prev_top:
                            bar += 1
                            staff = 1
                        else:
                            staff += 1
                        prev_top = measure.top
                        image = Image.open(images_path / 'system_{}_measure_{}.png'.format(j + 1, k + 1))
                        score_measures = []
                        mapping_page = next((p for p in score_mapping if p.pagenumber == page_nr))
                        mapping_system = next((s for s in mapping_page.systems if s.systemnumber == system_nr))
                        mapping_staff = next((s for s in mapping_system.staffs if s.staffnumber == staff))
                        measure_idx = mapping_system.measurestart + bar - 1
                        staff_parts = [part for part in parts if part.id in [p.id for p in mapping_staff.parts]]
                        for part in staff_parts:
                            score_measures.append(part.measure(measure_idx, collect=[], indicesNotNumbers=True))
                        self.measures.append(ClusterMeasure(measure, image, page_offset + page_nr, system_nr, bar, staff, part_nr, score_measures))
            # Use measures_path dir length here because images_path dir length can have missing pages.
            page_offset += len(list(measures_path.iterdir()))
        self.measures.sort(key=attrgetter('page', 'system', 'bar', 'staff'))
        for j, measure in enumerate(self.measures):
            measure.idx = j
            measure.label = self.labels[j]

    def load_score_mappings(self):
        score_mappings = []
        for path in self.score_mapping_path:
            with open(path) as f:
                score_mappings.append([ScoreMappingPage.from_json(json_data) for json_data in json.load(f)['pages']])
        self.score_mapping = score_mappings

    def load_scores(self):
        parts = []
        for path in self.score_path:
            parts.append(list(converter.parse(path).recurse(classFilter='Part')))
        self.parts = parts

    def load(self):
        self.labels = np.load(self.labels_path).tolist()
        self.medoids = np.load(self.medoids_path).tolist()
        self.load_score_mappings()
        self.load_scores()
        self.load_measures()


if __name__ == '__main__':
    # score = musicdata_dir / 'bach_brandenburg_concerto_5_part_1'
    score = musicdata_dir / 'beethoven_symphony_4'
    dirs = get_score_dirs(score)
    evaluator = ClusterEvaluator(dirs)
    evaluator.load()

import json
import re
import sys
import time
from itertools import groupby, product
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from operator import attrgetter
from sklearn_extra.cluster import KMedoids


class Measure:

    def __init__(self, start, top, end, bottom, system):
        self.start = start
        self.top = top
        self.end = end
        self.bottom = bottom
        self.system = system

    @staticmethod
    def from_json(json_data):
        return Measure(json_data['start'], json_data['top'], json_data['end'], json_data['bottom'], json_data['system'])

    def to_json(self):
        return {'start': self.start, 'top': self.top, 'end': self.end, 'bottom': self.bottom, 'system': self.system}


class MeasureProfile:

    def __init__(self, measure, image, page, system, bar, staff):
        self.measure = measure
        self.profile = np.asarray(ImageOps.invert(image).convert('1')).sum(axis=0)
        self.page = page
        self.system = system
        self.bar = bar
        self.staff = staff

    def __str__(self):
        return 'Page: {}\tSystem: {}\tBar: {}\tStaff: {}'.format(self.page, self.system, self.bar, self.staff)


class MeasureImage(MeasureProfile):

    def __init__(self, measure, image, page, system, bar, staff):
        super(MeasureImage, self).__init__(measure, image, page, system, bar, staff)
        self.image = image


class MeasureClusterer:

    def __init__(self, measures_path, images_path, dist_matrix_path):
        if isinstance(measures_path, list) != isinstance(images_path, list):
            raise ValueError('Params `measures_path` and `images_path` need to be of same type')
        if isinstance(measures_path, list):
            if len(measures_path) != len(images_path):
                raise ValueError('Params `measures_path` and `images_path` need to be of same length')
            self.measures_path = [Path(path) for path in measures_path]
            self.images_path = [Path(path) for path in images_path]
        else:
            self.measures_path = [Path(measures_path)]
            self.images_path = [Path(images_path)]
        self.dist_matrix_path = Path(dist_matrix_path)
        self.measure_images = None
        self.dist_matrix = None
        self.clusters = None
        self.labels = None
        self.medoids = None
        self.cluster_save_dir = None

    def load_measure_images(self, store_images=True):
        print('Loading measures...')
        self.measure_images = []
        page_offset = 0
        for measures_path, images_path in zip(self.measures_path, self.images_path):
            if not measures_path.exists():
                raise FileNotFoundError('No measures folder found at: {}'.format(self.measures_path))
            if not images_path.exists():
                raise FileNotFoundError('No images folder found at: {}'.format(self.images_path))
            for images_path in sorted(images_path.iterdir()):
                page_path = measures_path / (images_path.stem + '.json')
                if not images_path.exists():
                    raise FileNotFoundError('No image found at: {}'.format(images_path))
                page = page_offset + int(page_path.stem.split('_')[1])
                with open(page_path) as f:
                    measures = [Measure.from_json(json_data) for json_data in json.load(f)['measures']]
                measures.sort(key=attrgetter('system', 'start', 'top'))
                systems = [list(v) for k, v in groupby(measures, key=attrgetter('system'))]
                for i, system in enumerate(systems):
                    bar = 0
                    staff = 0
                    prev_top = system[0].top + 1
                    for j, measure in enumerate(system):
                        if measure.top < prev_top:
                            bar += 1
                            staff = 1
                        else:
                            staff += 1
                        prev_top = measure.top
                        image = Image.open(images_path / 'system_{}_measure_{}.png'.format(i + 1, j + 1))
                        measure_cls = MeasureImage if store_images else MeasureProfile
                        self.measure_images.append(measure_cls(measure, image, page, i + 1, bar, staff))
            # Use measures_path dir length here because images_path dir length can have missing pages.
            page_offset += len(list(measures_path.iterdir()))
        self.measure_images.sort(key=attrgetter('page', 'system', 'bar', 'staff'))
        for i, measure in enumerate(self.measure_images):
            measure.idx = i

    def normalize_dist_matrix(self, normalization=None):
        if normalization is None:
            return
        lengths = np.array([m.profile.shape[0] for m in self.measure_images])
        norm_matrix = list(product(lengths, repeat=2))
        if normalization == 'smallest':
            norm_matrix = np.min(norm_matrix, axis=1).reshape(-1, len(lengths))
        if normalization == 'largest':
            norm_matrix = np.max(norm_matrix, axis=1).reshape(-1, len(lengths))
        self.dist_matrix = self.dist_matrix / norm_matrix

    def get_distance_matrix(self, normalization=None):
        if self.dist_matrix_path is None or not self.dist_matrix_path.exists():
            raise FileNotFoundError('Could not find dist_matrix at {}'.format(self.dist_matrix_path))
        self.dist_matrix = np.load(self.dist_matrix_path)
        self.normalize_dist_matrix(normalization)

    def cluster_worker(self, k):
        start = time.time()
        c = KMedoids(n_clusters=k, method='pam', metric='precomputed').fit(self.dist_matrix)
        if self.cluster_save_dir is not None:
            np.save(self.cluster_save_dir / 'cluster_labels_{}.npy'.format(k), c.labels_)
            np.save(self.cluster_save_dir / 'cluster_medoids_{}.npy'.format(k), c.medoid_indices_)
            np.save(self.cluster_save_dir / 'cluster_inertia_{}.npy'.format(k), c.inertia_)
        end = time.time()
        print('Clustered {} clusters in {}s'.format(k, end - start), flush=True)

    def cluster(self, start=2, end=50, save_dir=None):
        print('Clustering based on distance matrix...', flush=True)
        ks = list(range(start, end + 1))
        self.cluster_save_dir = Path(save_dir)
        if not self.cluster_save_dir.exists():
            raise FileNotFoundError('Could not find save_dir at {}'.format(self.cluster_save_dir))
        done = sorted([int(re.search(r'\d+', s.stem).group()) for s in save_dir.glob('cluster_labels_*')])
        ks = [k for k in ks if k not in done and (k <= 100 or k % 5 == 0)]
        print(ks, flush=True)
        for k in ks:
            self.cluster_worker(k)


def get_parts(score):
    return sorted([part for part in score.iterdir() if part.is_dir() and part.name.startswith('part_')])


if __name__ == '__main__':
    score = Path(sys.argv[1])
    start = 2
    end = 150
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
    measures_path = score / 'measures'
    images_path = score / 'measure_images'
    cluster_save = score / 'clusters'
    cluster_save.mkdir(exist_ok=True)
    parts = get_parts(score)
    if len(parts) > 0:
        measures_path = [score / part.name / 'measures' for part in parts]
        images_path = [score / part.name / 'measure_images' for part in parts]
    dist_matrix_path = score / 'dist_matrix.npy'
    clusterer = MeasureClusterer(measures_path, images_path, dist_matrix_path)
    # Set store_images to False when only profiles are needed for memory efficiency
    clusterer.load_measure_images(store_images=False)
    clusterer.get_distance_matrix(normalization='smallest')
    clusterer.cluster(save_dir=cluster_save, start=start, end=end)

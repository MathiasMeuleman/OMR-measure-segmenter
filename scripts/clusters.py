import argparse
import json
import os
import re
import time
from itertools import groupby, product
from multiprocessing import Process, Queue, current_process
from pathlib import Path

from kmedoids import fastpam1, fasterpam
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


class Clustering:
    def __init__(self, medoids, labels, loss):
        self.medoids = medoids
        self.labels = labels
        self.loss = loss


def load_measure_images(measures_path, images_path):
    measure_images = []
    page_offset = 0
    for measures_path, images_path in zip(measures_path, images_path):
        if not measures_path.exists():
            raise FileNotFoundError('No measures folder found at: {}'.format(measures_path))
        if not images_path.exists():
            raise FileNotFoundError('No images folder found at: {}'.format(images_path))
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
                    measure_images.append(MeasureProfile(measure, image, page, i + 1, bar, staff))
        # Use measures_path dir length here because images_path dir length can have missing pages.
        page_offset += len(list(measures_path.iterdir()))
    measure_images.sort(key=attrgetter('page', 'system', 'bar', 'staff'))
    for i, measure in enumerate(measure_images):
        measure.idx = i
    return measure_images


def normalize_dist_matrix(dist_matrix, measure_images, normalization=None):
    if normalization is None:
        return
    lengths = np.array([m.profile.shape[0] for m in measure_images])
    norm_matrix = list(product(lengths, repeat=2))
    if normalization == 'smallest':
        norm_matrix = np.min(norm_matrix, axis=1).reshape(-1, len(lengths))
    if normalization == 'largest':
        norm_matrix = np.max(norm_matrix, axis=1).reshape(-1, len(lengths))
    return dist_matrix / norm_matrix


def get_distance_matrix(dist_matrix_path, measure_images, normalization=None):
    if dist_matrix_path is None or not dist_matrix_path.exists():
        raise FileNotFoundError('Could not find dist_matrix at {}'.format(dist_matrix_path))
    dist_matrix = np.load(dist_matrix_path)
    return normalize_dist_matrix(dist_matrix, measure_images, normalization)


def cluster_queue_worker(queue, method, measures_path, images_path, dist_matrix_path, cluster_save_dir, repetitions=10):
    name = current_process().name
    print('[{}] Initializing...'.format(name), flush=True)
    measure_images = load_measure_images(measures_path, images_path)
    print('[{}] Loaded measures'.format(name), flush=True)
    dist_matrix = get_distance_matrix(dist_matrix_path, measure_images, normalization='smallest').astype(np.float32)
    print('[{}] Loaded distance matrix'.format(name), flush=True)
    while True:
        k = queue.get()
        if k == '__DONE__':
            print('[{}] Received DONE; terminating'.format(name), flush=True)
            break
        k = int(k)
        print('[{}] Clustering around {} medoids'.format(name, k), flush=True)
        start = time.time()
        best_clustering = None
        if method == 'pam':
            c = KMedoids(n_clusters=k, method='pam', metric='precomputed').fit(dist_matrix)
            best_clustering = Clustering(c.medoid_indices_, c.labels_, c.inertia_)
        elif method == 'fastpam':
            for i in range(repetitions):
                c = fastpam1(dist_matrix, medoids=k)
                clustering = Clustering(c.medoids, c.labels, c.loss)
                if best_clustering is None or clustering.loss < best_clustering.loss:
                    best_clustering = clustering
        elif method == 'fasterpam':
            for i in range(repetitions):
                c = fasterpam(dist_matrix, medoids=k)
                clustering = Clustering(c.medoids, c.labels, c.loss)
                if best_clustering is None or clustering.loss < best_clustering.loss:
                    best_clustering = clustering
        else:
            raise ValueError('Method {} is not supported'.format(method))
        np.save(cluster_save_dir / '{}_cluster_labels_{}.npy'.format(method, k), best_clustering.labels)
        np.save(cluster_save_dir / '{}_cluster_medoids_{}.npy'.format(method, k), best_clustering.medoids)
        np.save(cluster_save_dir / '{}_cluster_inertia_{}.npy'.format(method, k), best_clustering.loss)
        end = time.time()
        print('[{}] Done clustering around {} medoids in {}s'.format(name, k, end-start), flush=True)


def get_parts(score):
    return sorted([part for part in score.iterdir() if part.is_dir() and part.name.startswith('part_')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('score')
    parser.add_argument('-p', action='store_true', default=False)
    parser.add_argument('--start', type=int, default=2)
    parser.add_argument('--end', type=int, default=150)
    parser.add_argument('--method', default='fastpam')
    args = parser.parse_args()

    score = Path(args.score)
    cluster_save = score / 'clusters'
    cluster_save.mkdir(exist_ok=True)
    if args.p:
        threads = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 2
    else:
        threads = 1
    parts = get_parts(score)
    if len(parts) > 0:
        measures_path = [score / part.name / 'measures' for part in parts]
        images_path = [score / part.name / 'measure_images' for part in parts]
    else:
        measures_path = [score / 'measures']
        images_path = [score / 'measure_images']
    dist_matrix_path = score / 'dist_matrix.npy'

    print('Clustering based on distance matrix...', flush=True)
    ks = list(range(args.start, args.end + 1))
    done = sorted([int(re.search(r'\d+', s.stem).group()) for s in cluster_save.glob('{}_cluster_labels_*'.format(args.method))])
    ks = [k for k in ks if k not in done and (k <= 100 or k % 5 == 0)]
    print(ks, flush=True)

    queue = Queue()
    processes = [Process(target=cluster_queue_worker, args=(queue, args.method, measures_path, images_path, dist_matrix_path, cluster_save)) for _ in range(threads)]

    for k in ks:
        queue.put(k)
    for p in processes:
        queue.put('__DONE__')

    for p in processes:
        p.start()

    for p in processes:
        p.join()


import json
import os
import sys
import time
from itertools import groupby
from multiprocessing import Pool

from PIL import Image, ImageOps
from fastdtw import fastdtw
from operator import attrgetter
import numpy as np
from pathlib import Path


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


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MeasureClusterer:

    def __init__(self, measures_path, images_path, dist_matrix_path, dist_matrix_save_path):
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
        self.dist_matrix_save_path = Path(dist_matrix_save_path)
        self.measure_images = None
        self.dist_matrix = None
        self.clusters = None

    def load_measure_images(self):
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
                        measure_cls = MeasureProfile
                        self.measure_images.append(measure_cls(measure, image, page, i + 1, bar, staff))
            # Use measures_path dir length here because images_path dir length can have missing pages.
            page_offset += len(list(measures_path.iterdir()))
        self.measure_images.sort(key=attrgetter('page', 'system', 'bar', 'staff'))


    def _get_measure_distance(self, profile1, profile2):
        dist, path = fastdtw(profile1, profile2, radius=20)
        return dist

    def get_measure_distance(self, measure1, measure2):
        return self._get_measure_distance(measure1.profile, measure2.profile)

    def distance_worker(self, args):
        i, j = args[0:2]
        dist = self.get_measure_distance(self.measure_images[i], self.measure_images[j])
        return i, j, dist

    def compute_distance_matrix_pool(self, max_threads, parts=None, mem_bench=False):
        dist_matrix_exists = self.dist_matrix_path.exists()
        parts_mode = parts is not None
        chunksize_per_thread = 30000
        chunksize = chunksize_per_thread * max_threads

        print('Building jobs queue with params:\n\tthreads: {}\n\tchunksize: {}\n\tchunksize_per_thread: {}\n'.format(max_threads, chunksize, chunksize_per_thread))

        if dist_matrix_exists:
            dist_matrix = np.load(self.dist_matrix_path).astype(np.half)
        else:
            dist_matrix = np.zeros((len(self.measure_images), len(self.measure_images)), dtype=np.half)

        tril = np.array(np.tril_indices_from(dist_matrix, k=1), dtype=np.int32)
        i = np.flatnonzero(dist_matrix[tril[0], tril[1]] == 0)
        args = tril.T[i]
        if int(len(args) / chunksize) == 0:
            args_list = np.array([args])
        else:
            args_list = np.array_split(args, int(len(args) / chunksize))

        part_start = 0
        part_end = len(args_list)
        if parts_mode:
            part_len = int(len(args_list) / parts[1])
            part_start = (parts[0] - 1) * part_len
            part_end = min(part_start + part_len, len(args_list))
        selected_args = args_list[part_start:part_end]

        if mem_bench:
            if not dist_matrix_exists:
                print('Mem bench:...')
                dist_matrix[:] = np.random.rand(*dist_matrix.shape)
                time.sleep(10)
                print(dist_matrix[0, 0])

        else:
            print('Needs to process {} batches'.format(len(args_list)))
            if parts_mode:
                print('Only processing part {}/{}, {} batches'.format(parts[0], parts[1], len(selected_args)))

            for idx, lst in enumerate(selected_args):
                start = time.time()
                pool = Pool(processes=max_threads)
                print('Processing idx {}, length {}'.format(idx, len(lst)))
                results = pool.map(self.distance_worker, lst)
                for (i, j, dist) in results:
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                np.save(self.dist_matrix_save_path, dist_matrix)
                pool.close()
                pool.join()
                end = time.time()
                print('\tDone in {}s'.format(round(end - start)))

        return dist_matrix


"""
This script is optimized for the high performance compute clusters on which the distance calculations were run.
"""
if __name__ == '__main__':
    part = Path(sys.argv[1])
    part_of_total = None
    mem_bench = False
    if len(sys.argv) >= 3:
        parts_str = sys.argv[2]
        if '/' in parts_str:
            part_of_total = (int(parts_str.split('/')[0]), int(parts_str.split('/')[1]))
        elif parts_str == 'mem-bench':
            mem_bench = True
        else:
            raise ValueError('Do not know what to do with arg "{}"'.format(parts_str))
    print('Calculating dist matrix for {}'.format(part))
    start = time.time()
    parts = [p for p in part.iterdir() if p.is_dir() and p.stem.startswith('part_')]
    if len(parts) > 0:
        measures_path = [p / 'measures' for p in parts]
        images_path = [p / 'measure_images' for p in parts]
    else:
        measures_path = part / 'measures'
        images_path = part / 'measure_images'
    dist_matrix_path = part / 'dist_matrix.npy'
    save_path = dist_matrix_path
    if part_of_total is not None:
        save_path = dist_matrix_path.parent / 'dist_matrix_{}.{}.npy'.format(part_of_total[0], part_of_total[1])
    threads = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 4

    clusterer = MeasureClusterer(measures_path, images_path, dist_matrix_path, save_path)
    clusterer.load_measure_images()
    dist_matrix = clusterer.compute_distance_matrix_pool(max_threads=threads, parts=part_of_total, mem_bench=mem_bench)
    if not mem_bench:
        np.save(save_path, dist_matrix)
    end = time.time()
    print('\tDone in {}'.format(end - start))

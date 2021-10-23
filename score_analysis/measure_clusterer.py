import json
import time
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from fastdtw import fastdtw
from kmedoids import fasterpam
from operator import attrgetter
from tqdm import tqdm

from score_analysis.measure_detector import Measure
from util.dirs import data_dir, get_musicdata_scores


class MeasureProfile:

    def __init__(self, measure, image, page, system, bar, staff):
        self.measure = measure
        # self.image = image
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


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def display_measure_images(measures):
    border_width = 4
    row_size = 10
    avg_width = sum([m.image.width for m in measures]) // len(measures)
    rows = list(chunks(measures, row_size))
    row_heights = []
    for row in rows:
        row_heights.append(max([int(min(m.image.width / avg_width, 1) * m.image.height) for m in row]))
    image = Image.new('L', (avg_width * row_size + border_width * 2 * row_size, sum(row_heights) + border_width * 2 * len(row_heights)), 255)
    for i, row in enumerate(rows):
        top = sum(row_heights[0:i]) + border_width * 2 * i
        for j, measure in enumerate(row):
            resize = min(measure.image.width / avg_width, 1)
            measure_img = ImageOps.expand(measure.image.resize((int(measure.image.width * resize), int(measure.image.height * resize))), border=border_width, fill='red')
            image.paste(measure_img, (j * avg_width + border_width * 2 * j, top))
    return image


class MeasureClusterer:

    def __init__(self, measures_path, images_path):
        self.measures_path = Path(measures_path)
        self.images_path = Path(images_path)
        self.intermediate_dist_matrix_path = self.measures_path.parent / 'intermediate_dist_matrix.npy'
        self.intermediate_metadata_path = self.measures_path.parent / 'intermediate_metadata.txt'
        self.measure_images = None
        self.dist_matrix = None
        self.clusters = None

    def load_measure_images(self, store_images=True):
        if not self.measures_path.exists():
            raise FileNotFoundError('No measures folder found at: {}'.format(self.measures_path))
        if not self.images_path.exists():
            raise FileNotFoundError('No images folder found at: {}'.format(self.images_path))
        self.measure_images = []
        for images_path in sorted(self.images_path.iterdir()):
            page_path = self.measures_path / (images_path.stem + '.json')
            if not images_path.exists():
                raise FileNotFoundError('No image found at: {}'.format(images_path))
            page = page_path.stem.split('_')[1]
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
        self.measure_images.sort(key=attrgetter('page', 'system', 'bar', 'staff'))

    def _get_measure_distance(self, profile1, profile2):
        dist, path = fastdtw(profile1, profile2, radius=20)
        return dist

    def get_measure_distance(self, measure1, measure2):
        return self._get_measure_distance(measure1.profile, measure2.profile)

    def distance_worker(self, args):
        i, js, save = args
        dists = []
        for j in js:
            dist = self.get_measure_distance(self.measure_images[i], self.measure_images[j])
            dists.append(dist)
        return i, js, dists

    def compute_distance_matrix_pool(self, max_threads, job_size, chunk_size, start_idx=0):
        print('Building jobs queue...')
        args_list = []
        for i in range(len(self.measure_images)):
            for start in range(0, i, job_size):
                end = min(start + job_size, len(self.measure_images))
                args_list.append((i, range(start, end), i % 10 == 0 and start == 0))
        args_list = list(chunks(args_list, chunk_size))

        print('Jobs queue built. {} jobs'.format(sum([len(l) for l in args_list])))
        print('Processing started with {} workers'.format(max_threads))
        print('Needs to process {} rows'.format(len(self.measure_images)))

        dist_matrix = np.zeros((len(self.measure_images), len(self.measure_images)))
        for idx, lst in enumerate(args_list):
            if idx <= start_idx:
                continue
            pool = Pool(processes=max_threads)
            print('Processing idx {}, length {}'.format(idx, len(lst)))
            results = pool.imap_unordered(self.distance_worker, lst, chunksize=max(1, int(chunk_size / max_threads)))
            for (i, js, dists) in results:
                for k, j in enumerate(js):
                    dist_matrix[i, j] = dists[k]
                    dist_matrix[j, i] = dists[k]
            np.save(self.intermediate_dist_matrix_path, dist_matrix)
            with open(self.intermediate_metadata_path, 'w') as f:
                f.write('{}'.format(idx))
            pool.close()
            pool.join()

        self.intermediate_dist_matrix_path.unlink()
        self.intermediate_metadata_path.unlink()
        return dist_matrix

    # def distance_matrix_worker(self, jobs, results):
    #     while not jobs.empty():
    #         (i, js) = jobs.get()
    #         dists = []
    #         for j in js:
    #             dist = self.get_measure_distance(self.measure_images[i], self.measure_images[j])
    #             dists.append(dist)
    #         results.put((i, js, dists))
    #         jobs.task_done()
    #
    # def compute_distance_matrix(self, max_threads, job_size):
    #     jobs = Queue()
    #     results = Queue()
    #     print('Building jobs queue...')
    #     for i in range(len(self.measure_images)):
    #         for start in range(0, i, job_size):
    #             end = min(start + job_size, len(self.measure_images))
    #             jobs.put((i, range(start, end)))
    #
    #     print('Jobs queue built. {} jobs'.format(jobs.qsize()))
    #     print('Processing started with {} workers'.format(max_threads))
    #     for i in range(max_threads):
    #         worker = Thread(target=self.distance_matrix_worker, args=(jobs, results))
    #         worker.start()
    #
    #     jobs.join()
    #
    #     print('Processing done, building distance matrix.')
    #     dist_matrix = np.zeros((len(self.measure_images), len(self.measure_images)))
    #     while not results.empty():
    #         (i, js, dists) = results.get()
    #         for k, j in enumerate(js):
    #             dist_matrix[i, j] = dists[k]
    #             dist_matrix[j, i] = dists[k]
    #     return dist_matrix

    def get_distance_matrix(self, matrix_path=None, max_threads=8, job_size=50, chunk_size=50):
        if matrix_path is None or not Path(matrix_path).exists():
            start_idx = 0
            if Path(self.intermediate_metadata_path).exists():
                with open(self.intermediate_metadata_path) as f:
                    start_idx = int(f.readline())
            self.dist_matrix = self.compute_distance_matrix_pool(max_threads=max_threads, job_size=job_size,
                                                                 chunk_size=chunk_size, start_idx=start_idx)
            print('tralalalalaa')
            if matrix_path is not None:
                np.save(matrix_path, self.dist_matrix)
        else:
            self.dist_matrix = np.load(Path(matrix_path))

    def cluster(self):
        c = fasterpam(self.dist_matrix, 10)
        for i in range(len(self.measure_images)):
            self.measure_images[i].label = c.labels[i]
        measures = sorted(self.measure_images, key=attrgetter('label'))
        self.clusters = sorted([list(v) for k, v in groupby(measures, key=attrgetter('label'))], key=lambda l: len(l), reverse=True)

    def _get_profile_image(self, measure):
        profile_img = np.ones((measure.image.height, measure.image.width)) * 255
        for i in range(measure.profile.size):
            profile_img[(profile_img.shape[0] - 1 - measure.profile[i]):, i] = 0
        return Image.fromarray(profile_img)

    def visualize_cluster(self, cluster):
        display_measure_images(cluster)

    def compare_measures_visual(self, idx1, idx2):
        measure1 = self.measure_images[idx1]
        measure2 = self.measure_images[idx2]
        dist = self.get_measure_distance(measure1, measure2)
        width = measure1.image.width + measure2.image.width + 100
        height = max(measure1.image.height, measure2.image.height) * 2

        image = Image.new('L', (width, height), 255)
        image.paste(measure1.image, (0, 0))
        image.paste(measure2.image, (measure1.image.width + 100, 0))

        profile1_img = self._get_profile_image(measure1)
        image.paste(profile1_img, (0, image.height - profile1_img.height))
        profile2_img = self._get_profile_image(measure2)
        image.paste(profile2_img, (image.width - profile2_img.width, image.height - profile2_img.height))

        draw = ImageDraw.Draw(image)
        draw.text((profile1_img.width + 50, image.height - 100), '{}'.format(dist), anchor='ms')
        image.show()


if __name__ == '__main__':
    for part in get_musicdata_scores(sort_reverse=True):
        print('Calculating dist matrix for {}'.format(part))
        start = time.time()
        measures_path = part / 'measures'
        images_path = part / 'measure_images'
        dist_matrix_path = part / 'dist_matrix.npy'

        clusterer = MeasureClusterer(measures_path, images_path)
        clusterer.load_measure_images(store_images=False)
        clusterer.get_distance_matrix(dist_matrix_path, max_threads=12, job_size=100, chunk_size=3)
        end = time.time()
        print('\tDone in {}'.format(end - start))

import json
import time
from itertools import groupby, product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from fastdtw import fastdtw
from kmedoids import fasterpam
from operator import attrgetter
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from score_analysis.measure_detector import Measure
from util.dirs import data_dir, get_musicdata_scores, musicdata_dir, get_parts


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


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MeasureCluster:

    def __init__(self, measures, label, medoid):
        self.measures = measures
        self.label = label
        self.medoid = medoid



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

    def _get_measure_distance(self, profile1, profile2):
        dist, path = fastdtw(profile1, profile2, radius=20)
        return dist

    def get_measure_distance(self, measure1, measure2):
        return self._get_measure_distance(measure1.profile, measure2.profile)

    def distance_worker(self, args):
        i, js = args
        dists = []
        for j in js:
            dist = self.get_measure_distance(self.measure_images[i], self.measure_images[j])
            dists.append(dist)
        return i, js, dists

    def compute_distance_matrix_pool(self, max_threads):
        dist_matrix_exists = self.dist_matrix_path.exists()
        chunksize_per_thread = 1000
        chunksize = chunksize_per_thread * max_threads

        print('Building jobs queue...')

        if dist_matrix_exists:
            dist_matrix = np.load(self.dist_matrix_path).astype(np.half)
        else:
            dist_matrix = np.zeros((len(self.measure_images), len(self.measure_images)), dtype=np.half)

        args_list = []
        for i in range(len(self.measure_images)):
            js = [j for j in range(i, len(self.measure_images)) if j != i and dist_matrix[i, j] == 0.0]
            for chunk in chunks(js, max_threads):
                args_list.append((i, chunk))
        args_list = list(chunks(args_list, chunksize))

        print('Jobs queue built. {} jobs'.format(sum([len(l) for l in args_list])))
        print('Processing started with {} workers'.format(max_threads))
        print('Needs to process {} rows'.format(len(self.measure_images)))

        dist_matrix = np.zeros((len(self.measure_images), len(self.measure_images)))
        for idx, lst in enumerate(args_list):
            start = time.time()
            pool = Pool(processes=max_threads)
            print('Processing idx {}, length {}'.format(idx, len(lst)))
            results = pool.map(self.distance_worker, lst)
            for (i, js, dists) in results:
                for k, j in enumerate(js):
                    dist_matrix[i, j] = dists[k]
                    dist_matrix[j, i] = dists[k]
            np.save(self.dist_matrix_path, dist_matrix)
            pool.close()
            pool.join()
            end = time.time()
            print('\tDone in {}s'.format(round(end - start)))

        self.dist_matrix = dist_matrix

    def normalize_dist_matrix(self, normalization=None):
        if normalization is None:
            return self.dist_matrix
        lengths = np.array([m.profile.shape[0] for m in self.measure_images])
        norm_matrix = list(product(lengths, repeat=2))
        if normalization == 'smallest':
            norm_matrix = np.min(norm_matrix, axis=1).reshape(-1, len(lengths))
        if normalization == 'largest':
            norm_matrix = np.max(norm_matrix, axis=1).reshape(-1, len(lengths))
        self.dist_matrix = self.dist_matrix / norm_matrix

    def get_distance_matrix(self, max_threads=8, normalization=None):
        if self.dist_matrix_path is None or not self.dist_matrix_path.exists():
            self.compute_distance_matrix_pool(max_threads=max_threads)
        else:
            self.dist_matrix = np.load(self.dist_matrix_path)
        self.normalize_dist_matrix(normalization)

    def cluster(self):
        print('Clustering based on distance matrix...')
        silhouette_scores = []
        clusterings = []
        float_dist_matrix = self.dist_matrix.astype(np.float32)
        for k in tqdm(range(2, 5)):
            c = fasterpam(float_dist_matrix, k)
            score = silhouette_score(float_dist_matrix, c.labels, metric='precomputed')
            clusterings.append(c)
            silhouette_scores.append(score)
        best_score_idx = max(range(len(silhouette_scores)), key=silhouette_scores.__getitem__)
        c = clusterings[best_score_idx]
        for i in range(len(self.measure_images)):
            self.measure_images[i].label = c.labels[i]
        measures = sorted(self.measure_images, key=attrgetter('label'))
        clustered_measures = sorted([list(v) for k, v in groupby(measures, key=attrgetter('label'))], key=lambda l: len(l), reverse=True)
        self.clusters = [MeasureCluster(measures, measures[0].label, c.medoids[measures[0].label]) for measures in clustered_measures]

    def _get_profile_image(self, measure):
        profile_img = np.ones((measure.image.height, measure.image.width)) * 255
        for i in range(measure.profile.size):
            profile_img[(profile_img.shape[0] - 1 - measure.profile[i]):, i] = 0
        return Image.fromarray(profile_img)

    def generate_cluster_images(self, cluster_idx, print_position=None, print_distances=False):
        cluster = self.clusters[cluster_idx]
        measures = cluster.measures
        border_width = 4
        row_size = 10
        avg_width = sum([m.image.width for m in measures]) // len(measures)
        rows = list(chunks(measures, row_size))
        image_width = avg_width * row_size + border_width * 2 * row_size
        max_height = int(1.41 * image_width)  # A4 ratios

        image_chunks = []
        image_chunk_heights = []

        chunk_rows = []
        chunk_row_heights = []

        for row in rows:
            row_height = max([int(min(m.image.width / avg_width, 1) * m.image.height) for m in row])
            if max_height > 0 and sum(chunk_row_heights) + row_height > max_height:
                image_chunks.append(chunk_rows)
                image_chunk_heights.append(chunk_row_heights)
                chunk_rows = []
                chunk_row_heights = []
            chunk_rows.append(row)
            chunk_row_heights.append(row_height)
        image_chunks.append(chunk_rows)
        image_chunk_heights.append(chunk_row_heights)

        medoid_img = self.measure_images[cluster.medoid].image
        header_height = medoid_img.height + border_width
        header_image = Image.new('L', (image_width, header_height), 255)
        header_image.paste(medoid_img, (int(image_width / 2 - medoid_img.width / 2), 0))

        images = []
        for rows, row_heights in zip(image_chunks, image_chunk_heights):
            image = ImageOps.pad(header_image.copy(), (image_width, sum(row_heights) + border_width * 2 * len(row_heights) + header_height), color=255, centering=(0, 0))
            for i, row in enumerate(rows):
                top = sum(row_heights[0:i]) + border_width * 2 * i + header_height
                for j, measure in enumerate(row):
                    resize = min(measure.image.width / avg_width, 1)
                    measure_img = ImageOps.expand(
                        measure.image.resize((int(measure.image.width * resize), int(measure.image.height * resize))),
                        border=border_width, fill='red')
                    position = (j * avg_width + border_width * 2 * j, top)
                    image.paste(measure_img, position)
                    draw = ImageDraw.Draw(image)
                    draw.text((int(image_width / 2 - medoid_img.width / 2 - 2 * border_width), 2 * border_width), str(cluster.medoid), anchor='rt')
                    draw_top = position[1] + 2 * border_width
                    if print_position is not None:
                        if print_position == 'idx':
                            text = str(measure.idx)
                        else:
                            text = '{}.{}.{}.{}'.format(measure.page, measure.system, measure.bar, measure.staff)
                        draw.text((position[0] + 2 * border_width, draw_top), text)
                    if print_distances:
                        text = str(round(self.dist_matrix[measure.idx, cluster.medoid], 2))
                        draw.text((position[0] + measure_img.width - 2 * border_width, draw_top), text, anchor='rt')
                    del draw
            images.append(image)
        return images

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
        return image

    def compare_measures(self, comparisons):
        images = []
        for (idx1, idx2) in comparisons:
            images.append(self.compare_measures_visual(idx1, idx2))
        row_size = 10
        gap_width = 5
        rows = list(chunks(images, row_size))
        row_dims = []
        for row in rows:
            width = sum([img.width for img in row]) + row_size * gap_width
            height = max([img.height for img in row]) + gap_width
            row_dims.append((width, height))
        image_width = max([dims[0] for dims in row_dims])
        image_height = sum([dims[1] for dims in row_dims])
        image = Image.new('L', (image_width, image_height), 255)
        curr_top = 0
        for i, row in enumerate(rows):
            curr_left = 0
            for img in row:
                image.paste(img, (curr_left, curr_top))
                curr_left += img.width + gap_width
            curr_top += row_dims[i][1]
        image.show()

if __name__ == '__main__':
    # score = musicdata_dir / 'bach_brandenburg_concerto_5_part_1'
    score = musicdata_dir / 'beethoven_symphony_1'
    measures_path = score / 'measures'
    images_path = score / 'measure_images'
    parts = get_parts(score)
    if len(parts) > 0:
        measures_path = [score / part.name / 'measures' for part in parts]
        images_path = [score / part.name / 'measure_images' for part in parts]
    dist_matrix_path = score / 'dist_matrix.npy'
    clusterer = MeasureClusterer(measures_path, images_path, dist_matrix_path)
    # Set store_images to false when calculating dist_matrix for memory efficiency
    clusterer.load_measure_images(store_images=True)
    # clusterer.compare_measures([(5077, 5099), (5098, 5099), (5133, 5136)])

    clusterer.get_distance_matrix(normalization='smallest')
    clusterer.cluster()
    for i in range(len(clusterer.clusters)):
        images = clusterer.generate_cluster_images(i, print_position='idx', print_distances=True)
        for j, image in enumerate(images):
            image.save('cluster_{}.{}.png'.format(i, j))

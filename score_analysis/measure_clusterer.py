import json
import re
import shutil
import time
from itertools import groupby, product
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from fastdtw import fastdtw
from kneed import KneeLocator
from operator import attrgetter
from sklearn_extra.cluster import KMedoids
from kmedoids import fastpam1, fasterpam
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
        self.cluster_save_dir = None
        self.measure_images = None
        self.dist_matrix = None
        self.clusters = None
        self.labels = None
        self.medoids = None
        self.ks = []
        self.all_labels = []
        self.all_medoids = []
        self.all_inertias = []

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

    class Clustering:
        def __init__(self, medoids, labels, loss):
            self.medoids = medoids
            self.labels = labels
            self.loss = loss

    def cluster(self, method='fastpam', start=2, end=50, save_dir=None):
        print('Clustering based on distance matrix...')
        self.ks = list(range(start, end + 1))
        clusterings = []
        float_dist_matrix = self.dist_matrix.astype(np.float32)
        for k in tqdm(self.ks):
            if method == 'pam':
                c = KMedoids(n_clusters=k, method='pam', metric='precomputed').fit(float_dist_matrix)
                clusterings.append(self.Clustering(c.medoid_indices_, c.labels_, c.inertia_))
            elif method == 'fastpam':
                c = fastpam1(float_dist_matrix, medoids=k)
                clusterings.append(self.Clustering(c.medoids, c.labels, c.loss))
            elif method == 'fasterpam':
                c = fasterpam(float_dist_matrix, medoids=k)
                clusterings.append(self.Clustering(c.medoids, c.labels, c.loss))
            else:
                raise ValueError('Method {} is not supported'.format(method))
        if save_dir is not None:
            save_dir = Path(save_dir)
            if save_dir.exists():
                for i in range(len(self.ks)):
                    c = clusterings[i]
                    np.save(save_dir / '{}_cluster_labels_{}.npy'.format(method, self.ks[i]), c.labels)
                    np.save(save_dir / '{}_cluster_medoids_{}.npy'.format(method, self.ks[i]), c.medoids)
                    np.save(save_dir / '{}_cluster_inertia_{}.npy'.format(method, self.ks[i]), c.loss)
        kneedle = KneeLocator(self.ks, [c.loss for c in clusterings], curve='convex', direction='decreasing')
        plt.style.use('ggplot')
        kneedle.plot_knee()
        plt.show()
        selected_clustering_idx = self.ks.index(kneedle.knee)
        print('Selected {} clusters'.format(self.ks[selected_clustering_idx]))
        c = clusterings[selected_clustering_idx]
        for i in range(len(self.measure_images)):
            self.measure_images[i].label = c.labels[i]
        self.labels = c.labels
        self.medoids = c.medoids
        measures = sorted(self.measure_images, key=attrgetter('label'))
        clustered_measures = [list(v) for k, v in groupby(measures, key=attrgetter('label'))]
        self.clusters = [MeasureCluster(measures, measures[0].label, c.medoids[measures[0].label]) for measures in clustered_measures]

    def load_clusters(self, save_dir, method='fastpam'):
        self.cluster_save_dir = Path(save_dir)
        self.ks = sorted([int(re.search(r'\d+', s.stem).group()) for s in self.cluster_save_dir.glob('{}_cluster_labels_*'.format(method))])
        self.ks = [k for k in self.ks if k <= 100 or k % 10 == 0]
        for k in self.ks:
            self.all_labels.append(np.load(self.cluster_save_dir / '{}_cluster_labels_{}.npy'.format(method, k)).tolist())
            self.all_medoids.append(np.load(self.cluster_save_dir / '{}_cluster_medoids_{}.npy'.format(method, k)).tolist())
            self.all_inertias.append(float(np.load(self.cluster_save_dir / '{}_cluster_inertia_{}.npy'.format(method, k))))

    def find_optimal_clustering(self, method='fastpam'):
        kneedle = KneeLocator(self.ks, self.all_inertias, curve='convex', direction='decreasing')
        plt.style.use('ggplot')
        kneedle.plot_knee()
        left, right = plt.gca().get_xlim()
        bottom, top = plt.gca().get_ylim()
        plt.title(str(self.dist_matrix_path.parent.name))
        plt.text(kneedle.elbow + (right - left) * 0.01, bottom + (top - bottom) * 0.01, 'Elbow point at: {}'.format(kneedle.elbow))
        plt.show()
        idx = self.ks.index(kneedle.elbow)
        self.labels = self.all_labels[idx]
        self.medoids = self.all_medoids[idx]
        for i in range(len(self.measure_images)):
            self.measure_images[i].label = self.labels[i]
        measures = sorted(self.measure_images, key=attrgetter('label'))
        clustered_measures = [list(v) for k, v in groupby(measures, key=attrgetter('label'))]
        self.clusters = [MeasureCluster(measures, measures[0].label, self.medoids[measures[0].label]) for measures in clustered_measures]
        shutil.copy(self.cluster_save_dir / '{}_cluster_medoids_{}.npy'.format(method, kneedle.elbow), self.cluster_save_dir.parent / 'cluster_medoids.npy')
        shutil.copy(self.cluster_save_dir / '{}_cluster_labels_{}.npy'.format(method, kneedle.elbow), self.cluster_save_dir.parent / 'cluster_labels.npy')

    def _get_profile_image(self, measure):
        profile_img = np.ones((measure.image.height, measure.image.width)) * 255
        for i in range(measure.profile.size):
            profile_img[(profile_img.shape[0] - 1 - measure.profile[i]):, i] = 0
        return Image.fromarray(profile_img)

    def generate_medoids_image(self):
        row_size = 7
        label_heigth = 30
        border_size = 4
        medoids = [self.measure_images[c.medoid].image for c in self.clusters]
        rows = list(chunks(medoids, row_size))
        row_heights = []
        row_widths = []
        for row in rows:
            row_heights.append(max([m.height for m in row]) + label_heigth)
            row_widths.append(sum([m.width + 2 * border_size for m in row]))
        image = Image.new('L', (max(row_widths), sum(row_heights)), 255)
        draw = ImageDraw.Draw(image)
        for i, row in enumerate(rows):
            top = sum(row_heights[0:i]) + i * label_heigth
            for j, medoid in enumerate(row):
                left = sum([m.width for m in row[0:j]]) + j * 2 * border_size
                image.paste(medoid, (left, top))
                draw.text((int(left + medoid.width // 2), top + medoid.height + border_size), str(i * row_size + j), anchor='ms')
        return image

    def generate_cluster_images(self, cluster_idx, approx=10, print_position=None, print_distances=False):
        Image.MAX_IMAGE_PIXELS = None  # Disable max size check
        cluster = self.clusters[cluster_idx]
        measures = cluster.measures
        border_width = 4
        image_width = sum(m.image.width for m in measures[0:approx]) + approx * border_width

        image = Image.new('L', (image_width, image_width), 255)
        medoid_img = next(m for m in self.measure_images if m.idx == cluster.medoid).image
        image.paste(medoid_img, (int(image_width / 2 - medoid_img.width / 2), 0))
        cur_x = 0
        cur_y = medoid_img.height
        largest_height = 0

        for measure in measures:
            measure_image = ImageOps.expand(measure.image, border=border_width)
            if cur_x + measure_image.width >= image.width:
                cur_x = 0
                cur_y += largest_height
                largest_height = 0
            if cur_y + measure_image.height >= image.height:
                image = ImageOps.pad(image, (image.width, image.height + image_width), color=255, centering=(0, 0))
            image.paste(measure_image, (cur_x, cur_y))
            cur_x += measure_image.width
            largest_height = max(largest_height, measure_image.height)

            draw = ImageDraw.Draw(image)
            draw_top = cur_y + 2 * border_width
            if print_position is not None:
                if print_position == 'idx':
                    text = str(measure.idx)
                else:
                    text = '{}.{}.{}.{}'.format(measure.page, measure.system, measure.bar, measure.staff)
                draw.text((cur_x - measure_image.width + 2 * border_width, draw_top), text)
            if print_distances:
                text = str(round(self.dist_matrix[measure.idx, cluster.medoid], 2))
                draw.text((cur_x - 2 * border_width, draw_top), text, anchor='rt')
            del draw

        return image.crop((0, 0, image.width, cur_y + largest_height))

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
    score = musicdata_dir / 'beethoven_symphony_5'
    measures_path = score / 'measures'
    images_path = score / 'measure_images'
    cluster_images = score / 'cluster_images'
    cluster_save = score / 'clusters'
    cluster_images.mkdir(exist_ok=True)
    cluster_save.mkdir(exist_ok=True)
    parts = get_parts(score)
    if len(parts) > 0:
        measures_path = [score / part.name / 'measures' for part in parts]
        images_path = [score / part.name / 'measure_images' for part in parts]
    dist_matrix_path = score / 'dist_matrix.npy'
    clusterer = MeasureClusterer(measures_path, images_path, dist_matrix_path)
    clusterer.load_measure_images(store_images=True)
    clusterer.load_clusters(save_dir=cluster_save)
    clusterer.find_optimal_clustering()
    for i in range(len(clusterer.clusters))[18:19]:
        image = clusterer.generate_cluster_images(i, print_position='idx', print_distances=False, approx=35)
        image.save(cluster_images / 'cluster_{}.png'.format(i))

    # for score in get_musicdata_scores(follow_parts=False):
    #     print(score)
    #     measures_path = score / 'measures'
    #     images_path = score / 'measure_images'
    #     cluster_images = score / 'cluster_images'
    #     cluster_save = score / 'clusters'
    #     cluster_images.mkdir(exist_ok=True)
    #     cluster_save.mkdir(exist_ok=True)
    #     parts = get_parts(score)
    #     if len(parts) > 0:
    #         measures_path = [score / part.name / 'measures' for part in parts]
    #         images_path = [score / part.name / 'measure_images' for part in parts]
    #     dist_matrix_path = score / 'dist_matrix.npy'
    #     clusterer = MeasureClusterer(measures_path, images_path, dist_matrix_path)
    #     clusterer.load_measure_images(store_images=True)
    #     clusterer.load_clusters(save_dir=cluster_save)
    #     clusterer.find_optimal_clustering()
    #     for f in cluster_images.glob('*'):
    #         f.unlink()
    #     for i in range(len(clusterer.clusters)):
    #         image = clusterer.generate_cluster_images(i, print_position='idx', print_distances=False)
    #         image.save(cluster_images / 'cluster_{}.png'.format(i))
    #     clusterer.generate_medoids_image().save(cluster_images / 'medoids.png')

    # Set store_images to false when calculating dist_matrix for memory efficiency
    # clusterer.load_measure_images(store_images=True)
    # clusterer.get_distance_matrix(normalization='smallest')
    # clusterer.compare_measures_visual(87, 16899).show()

    # clusterer.cluster(method='fastpam', save_dir=cluster_save)
    # clusterer.save_clusters(score)

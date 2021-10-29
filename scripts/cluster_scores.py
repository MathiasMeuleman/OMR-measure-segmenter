import numpy as np

from score_analysis.measure_clusterer import MeasureClusterer
from util.dirs import get_musicdata_scores, get_parts

if __name__ == '__main__':
    for score in get_musicdata_scores(follow_parts=False):
        print('Clustering {}...'.format(score.name))
        dist_matrix_path = score / 'dist_matrix.npy'
        if not dist_matrix_path.exists():
            print('Skipping {}: no dist_matrix'.format(score.name))
            continue
        dist_matrix = np.load(dist_matrix_path)
        if ((dist_matrix == 0).sum() - dist_matrix.shape[0]) > 0:
            print('Skipping {}: incomplete dist_matrix'.format(score.name))
            continue
        measures_path = score / 'measures'
        images_path = score / 'measure_images'
        parts = get_parts(score)
        if len(parts) > 0:
            measures_path = [score / part.name / 'measures' for part in parts]
            images_path = [score / part.name / 'measure_images' for part in parts]
        cluster_images = score / 'cluster_images'
        cluster_images.mkdir(exist_ok=True, parents=True)
        clusterer = MeasureClusterer(measures_path, images_path, dist_matrix_path)
        clusterer.load_measure_images()
        clusterer.get_distance_matrix()
        clusterer.cluster()
        for i, c in enumerate(clusterer.clusters):
            images = clusterer.visualize_cluster(c)
            for j, image in enumerate(images):
                image.save(cluster_images / 'cluster_{}.{}.png'.format(i, j))

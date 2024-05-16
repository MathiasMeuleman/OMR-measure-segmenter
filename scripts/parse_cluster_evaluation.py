import re

from util.dirs import get_musicdata_scores


def get_evaluation_medoid_score(evaluation):
    evaluation = evaluation.split('\n', 3)[-1]
    clusters = evaluation.split('\n\n')
    total_score = 0
    medoid_in_largest_group = 0
    total_clusters = 0
    for i, cluster in enumerate(clusters):
        total_clusters += 1
        cluster_size = int(re.findall(r'\d+', cluster.split('\n', 1)[0])[2])
        cluster = cluster.split('\n', 3)[-1]
        in_largest_group = 'x' in cluster.split('\n')[0]
        medoid_in_largest_group += in_largest_group
        group = next(l for l in cluster.split('\n') if 'x' in l)
        group_size = int(re.search(r'\d+', group).group())
        medoid_score = group_size / cluster_size
        total_score += medoid_score
    return total_score / len(clusters), medoid_in_largest_group, total_clusters


if __name__ == '__main__':
    total_clusters = 0
    medoid_in_largest_group = 0
    for score in get_musicdata_scores(follow_parts=False):
        with open(score / 'cluster_evaluation.txt') as f:
            total_score, in_largest_group, clusters = get_evaluation_medoid_score(f.read())
            total_clusters += clusters
            medoid_in_largest_group += in_largest_group
            print('{}\t{}'.format(score.name, total_score))
    print('{} / {} clusters have medoid in largest group'.format(medoid_in_largest_group, total_clusters))

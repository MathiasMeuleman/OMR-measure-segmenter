from itertools import groupby

from operator import attrgetter

from score_analysis.cluster_evaluator import ClusterEvaluator
from util.dirs import get_musicdata_scores, get_score_dirs
from util.table_generator import generate_table, TableColumn

if __name__ == '__main__':
    columns = [TableColumn('#Clusters', "clusters", "r"), TableColumn("#Classes", "classes", "r")]
    data = []
    for score in get_musicdata_scores(follow_parts=False):
        print(score.name)
        dirs = get_score_dirs(score)
        evaluator = ClusterEvaluator(dirs)
        evaluator.load()
        evaluator.encode_rhythms()
        sorted_measures = sorted(evaluator.measures, key=attrgetter('rhythm'))
        grouped_measures = [list(v) for k, v in groupby(sorted_measures, key=attrgetter('rhythm'))]
        data.append({'score': score.name, 'clusters': len(evaluator.medoids), 'classes': len(grouped_measures)})
    print(generate_table(columns, data))

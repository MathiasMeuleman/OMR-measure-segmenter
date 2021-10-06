import json

from score_analysis.measure_detector import Measure
from util.dirs import musicdata_dir


def get_measure_path_sort_key(path):
    return int(path.stem.split('_')[1])


def verify_musicdata_score(score_dir):
    combine_parts = sorted([d for d in score_dir.iterdir() if d.is_dir() and 'part_' in d.name])
    if len(combine_parts) > 0:
        annotations = []
        for part in combine_parts:
            with open(part / 'annotations.txt') as file:
                part_annotations = [list(map(lambda x: list(map(int, x.split(','))), line.rstrip().split(' '))) for line
                                    in file]
                annotations.extend(part_annotations)
    else:
        with open(score_dir / 'annotations.txt') as file:
            annotations = [list(map(lambda x: list(map(int, x.split(','))), line.rstrip().split(' '))) for line in file]
    if len(combine_parts) > 0:
        measure_paths = []
        for part in combine_parts:
            measure_paths.extend(sorted([file for file in (part / 'measures').iterdir()], key=get_measure_path_sort_key))
    else:
        measure_paths = sorted([file for file in (score_dir / 'measures').iterdir()], key=get_measure_path_sort_key)
    if len(measure_paths) != len(annotations):
        raise AssertionError(
            '{}: Found {} pages, expected {}'.format(score_dir.parts[-1], len(measure_paths), len(annotations)))
    true_avg_count = found_avg_count = 0
    page_accuracy_sum = 0
    page_results = []
    for i, path in enumerate(measure_paths):
        true_measures = sum([x[0] * x[1] for x in annotations[i]])
        true_avg_count += true_measures
        with open(path) as f:
            measure_collection = json.load(f)
            measures = [Measure.from_json(json_data) for json_data in measure_collection['measures']]
        found_measures = len(measures)
        found_avg_count += found_measures
        page_accuracy = 1 - (abs(true_measures - found_measures) / true_measures)
        page_accuracy_sum += page_accuracy
        page_results.append({'page': i + 1, 'found_measures': found_measures, 'true_measures': true_measures, 'page_accuracy': page_accuracy})
    accuracy = 1 - (abs(true_avg_count - found_avg_count) / true_avg_count)
    page_accuracy_avg = page_accuracy_sum / len(annotations)
    results = {
        'score': score_dir.parts[-1],
        'accuracy': accuracy,
        'avg_page_accuracy': page_accuracy_avg,
        'page_results': page_results,
    }
    return results


def print_results(results):
    print('Score {}'.format(results['score']))
    print('Page\tFound\tTrue\tAccuracy')
    for page in results['page_results']:
        if page['page_accuracy'] < 1:
            print('{}\t\t{}\t\t{}\t\t{}'.format(page['page'], page['true_measures'], page['found_measures'], page['page_accuracy']))
    print('Total accuracy {}, avg. page accuracy {}\n------------------------'.format(results['accuracy'], results['avg_page_accuracy']))


if __name__ == '__main__':
    for score in sorted([score for score in musicdata_dir.iterdir() if score.is_dir()]):
        results = verify_musicdata_score(score)
        print_results(results)


from tqdm import tqdm
from posixpath import join
from collections import namedtuple
from segmenter.old.image_util import data_dir
from util.files import get_sorted_page_paths

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


# def compare_results(detected_pages, true_pages):
#     outputs = []
#     detected_system_count = 0
#     detected_measure_count = 0
#     detected_staff_count = 0
#     total_systems = sum([len(p.systems) for p in true_pages])
#     total_measures = sum([sum([s.measures for s in p.systems]) for p in true_pages])
#     total_staffs = sum([sum([s.staffs for s in p.systems]) for p in true_pages])
#     for i in range(len(detected_pages)):
#         detected_system_count += len(detected_pages[i].systems)
#         if len(detected_pages[i].systems) != len(true_pages[i].systems):
#             outputs.append('Page {}: Expected {} systems, but got {}'.format(i, len(true_pages[i].systems), len(detected_pages[i].systems)))
#     if detected_system_count == total_systems:
#         for i in range(len(detected_pages)):
#             detected_systems, true_systems = detected_pages[i].systems, true_pages[i].systems
#             for j in range(len(detected_pages[i].systems)):
#                 detected_measure_count += len(detected_systems[j].measures)
#                 if len(detected_systems[j].measures) != true_systems[j].measures:
#                     outputs.append('Page {}, system {}: Expected {} measures, but got {}'.format(i, j, true_systems[j].measures, len(detected_systems[j].measures)))
#                 system_staffs = list(map(lambda k: len(detected_systems[j].measures[k].staffs), range(len(detected_systems[j].measures))))
#                 if len(set(system_staffs)) > 1:
#                     outputs.append('Page {}, system {}: Not all measures have equal numbers of staffs'.format(i, j))
#                 else:
#                     detected_staff_count += system_staffs[0]
#                     if system_staffs[0] != true_systems[j].staffs:
#                         outputs.append('Page {}, system {}: Expected {} staffs, but got {}'.format(i, j, true_systems[j].staffs, len(detected_systems[j].measures[0].staffs)))
#     print('Test results:\n=============')
#     print('System score: {}/{} ({}%)'.format(detected_system_count, total_systems, round(detected_system_count/total_systems * 100)))
#     print('Measure score: {}/{} ({}%)'.format(detected_measure_count, total_measures, round(detected_measure_count/total_measures * 100)))
#     print('Staff score: {}/{} ({}%)'.format(detected_staff_count, total_staffs, round(detected_staff_count/total_staffs * 100)))
#     print('\n'.join(outputs))

def compare_results(result_pages, true_pages, score_name):
    outputs = []
    if len(result_pages) == len(true_pages):
        for i, (true_page, result_page) in enumerate(zip(true_pages, result_pages)):
            if len(true_page.systems) == len(result_page.systems):
                for j, (true_system, result_system) in enumerate(zip(true_page.systems, result_page.systems)):
                    if true_system.staffs != result_system.staffs:
                        outputs.append('Staffs on page {}, system {} don\'t match: expected {} staffs, but found {} staffs'.format(i+1, j+1, true_system.staffs, result_system.staffs))
                    if true_system.measures != result_system.measures:
                        outputs.append('Measures on page {}, system {} don\'t match: expected {} measures, but found {} measures'.format(i + 1, j + 1, true_system.measures, result_system.measures))
            else:
                outputs.append('Systems on page {} don\'t match: expected {} systems, but found {} systems'.format(i+1, len(true_page.systems), len(result_page.systems)))
    else:
        outputs.append('Pages don\'t match: expected {} pages, but found {} pages'.format(len(true_pages), len(true_pages)))
    print('Test results {}:\n============='.format(score_name))
    print('\n'.join(outputs))
    print('\n')


def build_true_system(system_str):
    properties = list(map(int, system_str.split(',')))
    return TrueSystem(staffs=properties[0], measures=properties[1])


def evaluate(score_name):
    with open(join(data_dir, score_name, 'annotations_v2.txt')) as file:
        baseline = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    true_pages = list(map(lambda s: TruePage(systems=s), baseline))

    with open(join(data_dir, score_name, 'annotation_results_lines.txt')) as file:
        results = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    result_pages = list(map(lambda s: TruePage(systems=s), results))

    compare_results(result_pages, true_pages, score_name)


if __name__ == '__main__':
    scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l\'Apprenti_Sorcier', 'Haydn_Symphony_104_London', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31', 'Schubert_Symphony_4', 'Van_Bree_Allegro']
    for score in scores:
        evaluate(score)

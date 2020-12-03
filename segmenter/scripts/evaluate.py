from tqdm import tqdm
from posixpath import join
from collections import namedtuple
from segmenter.old.image_util import data_dir
from segmenter.measure_detector import detect_measures
from util.files import get_sorted_page_paths

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


def compare_results(detected_pages, true_pages):
    outputs = []
    detected_system_count = 0
    detected_measure_count = 0
    detected_staff_count = 0
    total_systems = sum([len(p.systems) for p in true_pages])
    total_measures = sum([sum([s.measures for s in p.systems]) for p in true_pages])
    total_staffs = sum([sum([s.staffs for s in p.systems]) for p in true_pages])
    for i in range(len(detected_pages)):
        detected_system_count += len(detected_pages[i].systems)
        if len(detected_pages[i].systems) != len(true_pages[i].systems):
            outputs.append('Page {}: Expected {} systems, but got {}'.format(i, len(true_pages[i].systems), len(detected_pages[i].systems)))
    if detected_system_count == total_systems:
        for i in range(len(detected_pages)):
            detected_systems, true_systems = detected_pages[i].systems, true_pages[i].systems
            for j in range(len(detected_pages[i].systems)):
                detected_measure_count += len(detected_systems[j].measures)
                if len(detected_systems[j].measures) != true_systems[j].measures:
                    outputs.append('Page {}, system {}: Expected {} measures, but got {}'.format(i, j, true_systems[j].measures, len(detected_systems[j].measures)))
                system_staffs = list(map(lambda k: len(detected_systems[j].measures[k].staffs), range(len(detected_systems[j].measures))))
                if len(set(system_staffs)) > 1:
                    outputs.append('Page {}, system {}: Not all measures have equal numbers of staffs'.format(i, j))
                else:
                    detected_staff_count += system_staffs[0]
                    if system_staffs[0] != true_systems[j].staffs:
                        outputs.append('Page {}, system {}: Expected {} staffs, but got {}'.format(i, j, true_systems[j].staffs, len(detected_systems[j].measures[0].staffs)))
    print('Test results:\n=============')
    print('System score: {}/{} ({}%)'.format(detected_system_count, total_systems, round(detected_system_count/total_systems * 100)))
    print('Measure score: {}/{} ({}%)'.format(detected_measure_count, total_measures, round(detected_measure_count/total_measures * 100)))
    print('Staff score: {}/{} ({}%)'.format(detected_staff_count, total_staffs, round(detected_staff_count/total_staffs * 100)))
    print('\n'.join(outputs))


def build_true_system(system_str):
    properties = list(map(int, system_str.split(',')))
    return TrueSystem(staffs=properties[0], measures=properties[1])


def evaluate(score_name):
    with open(join(data_dir, score_name, 'annotations.txt')) as file:
        baseline = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    true_pages = list(map(lambda s: TruePage(systems=s), baseline))

    page_paths = get_sorted_page_paths(join(data_dir, score_name, 'ppm-300'))
    if len(page_paths) != len(true_pages):
        print('Expected {} pages, but got {}'.format(len(true_pages), len(page_paths)))
        return

    pages = []
    for i, path in tqdm(enumerate(page_paths)):
        page = detect_measures(path)
        pages.append(page)
    compare_results(pages, true_pages)


if __name__ == '__main__':
    scores = ['Beethoven_Sextet']
    for score in scores:
        evaluate(score)

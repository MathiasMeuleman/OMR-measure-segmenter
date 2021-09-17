import json

from segmenter.measure_detector import Page, Staff, SystemMeasure, System, Measure
from segmenter.scripts.evaluation.evaluate import compare_results
from util.dirs import eval_dir


def build_fake_system(system_str):
    properties = list(map(int, system_str.split(',')))
    staffs = [Staff(ulx=0, uly=0, lrx=0, lry=0) for i in range(properties[0])]
    system_measures = [SystemMeasure(ulx=0, uly=0, lrx=0, lry=0) for i in range(properties[1])]
    measures = [Measure(ulx=0, uly=0, lrx=0, lry=0) for i in range(properties[0] * properties[1])]
    return System(staffs=staffs, system_measures=system_measures, measures=measures, ulx=0, uly=0, lrx=0, lry=0)


def build_system(system_data):
    staffs = [Staff(**staff_data) for staff_data in system_data['staffs']]
    system_measures = [SystemMeasure(**system_measure_data) for system_measure_data in system_data['system_measures']]
    measures = [Measure(**measure_data) for measure_data in system_data['measures']]
    system_bb = {k: system_data[k] for k in ('ulx', 'uly', 'lrx', 'lry')}
    return System(**system_bb, staffs=staffs, system_measures=system_measures, measures=measures)


def evaluate_dataset():
    version = 'test_json'
    # scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l_Apprenti_Sorcier',
    #           'Haydn_Symphony_104_London', 'Mahler_Symphony_1', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31',
    #           'Schubert_Symphony_4']
    scores = ['test']
    for score in scores:

        with open(eval_dir / 'truth' / '{}_annotations.txt'.format(score)) as file:
            baseline = [[build_fake_system(system_str) for system_str in line.rstrip().split(' ')] for line in file]
        true_pages = [Page(height=0, width=0, rotation=0, name='transcript_{}.png'.format(i+1), systems=s) for i, s in enumerate(baseline)]

        result_pages = []
        for page_file in sorted((eval_dir / version / 'annotations' / score).glob('*.json')):
            with open(page_file) as file:
                page_data = json.loads(file.read())
                systems = [build_system(system_data) for system_data in page_data['systems']]
                page_props = {k: page_data[k] for k in ('height', 'width', 'rotation')}
                result_pages.append(Page(**page_props, systems=systems, name=page_file.stem))

        compare_results(result_pages, true_pages, score, version)


if __name__ == '__main__':
    evaluate_dataset()

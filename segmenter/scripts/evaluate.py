from collections import namedtuple
from posixpath import join

from segmenter.dirs import eval_dir

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


def compare_results(result_pages, true_pages, score_name, version):
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
    output = 'Test results {}:\n============='.format(score_name)
    output += '\n' + '\n'.join(outputs) + '\n'
    print(output)
    with open(join(eval_dir, version, 'results', '{}_evaluation_results.txt').format(score_name), 'w') as file:
        file.write(output)


def build_true_system(system_str):
    properties = list(map(int, system_str.split(',')))
    return TrueSystem(staffs=properties[0], measures=properties[1])


def evaluate(score_name, version):
    with open(join(eval_dir, 'truth', '{}_annotations.txt'.format(score_name))) as file:
        baseline = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    true_pages = list(map(lambda s: TruePage(systems=s), baseline))

    with open(join(eval_dir, version, 'annotations', '{}_annotation_results.txt'.format(score_name))) as file:
        results = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    result_pages = list(map(lambda s: TruePage(systems=s), results))

    compare_results(result_pages, true_pages, score_name, version)


if __name__ == '__main__':
    version = 'current'
    scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l_Apprenti_Sorcier', 'Haydn_Symphony_104_London', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31', 'Schubert_Symphony_4', 'Van_Bree_Allegro']
    for score in scores:
        evaluate(score, version)

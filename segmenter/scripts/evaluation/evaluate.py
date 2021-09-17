from collections import namedtuple

from util.dirs import eval_dir

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


def compare_results(result_pages, true_pages, score_name, version):
    outputs = []
    total_systems = correct_systems = total_staffs = correct_staffs = total_system_measures = correct_system_measures = 0
    if len(result_pages) == len(true_pages):
        for i, (true_page, result_page) in enumerate(zip(true_pages, result_pages)):
            total_systems += 1
            if len(true_page.systems) == len(result_page.systems):
                correct_systems += 1
                for j, (true_system, result_system) in enumerate(zip(true_page.systems, result_page.systems)):
                    total_staffs += 1
                    total_system_measures += 1
                    if len(true_system.staffs) == len(result_system.staffs):
                        correct_staffs += 1
                    else:
                        outputs.append('Staffs on page {}, system {} don\'t match: expected {} staffs, but found {} staffs'.format(i+1, j+1, len(true_system.staffs), len(result_system.staffs)))
                    if len(true_system.system_measures) == len(result_system.system_measures):
                        correct_system_measures += 1
                    else:
                        outputs.append('System measures on page {}, system {} don\'t match: expected {} measures, but found {} measures'.format(i + 1, j + 1, len(true_system.system_measures), len(result_system.system_measures)))
            else:
                outputs.append('Systems on page {} don\'t match: expected {} systems, but found {} systems'.format(i+1, len(true_page.systems), len(result_page.systems)))
    else:
        outputs.append('Pages don\'t match: expected {} pages, but found {} pages'.format(len(true_pages), len(true_pages)))
    output = 'Test results {}:\n============='.format(score_name)
    output += '\n' + 'System detection score:\t' + str(round(correct_systems / total_systems, 4) * 100) + '%'
    output += '\n' + 'Staff detection score:\t' + str(round(correct_staffs / total_staffs, 4) * 100) + '%'
    output += '\n' + 'System measure detection score:\t' + str(round(correct_system_measures / total_system_measures, 4) * 100) + '%'
    output += '\n' + '\n'.join(outputs) + '\n'
    print(output)
    output_dir = eval_dir / version / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / '{}_evaluation_results.txt'.format(score_name), 'w') as file:
        file.write(output)


def build_true_system(system_str):
    properties = list(map(int, system_str.split(',')))
    return TrueSystem(staffs=properties[0], measures=properties[1])


def evaluate(score_name, version):
    with open(eval_dir / 'truth' / '{}_annotations.txt'.format(score_name)) as file:
        baseline = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    true_pages = list(map(lambda s: TruePage(systems=s), baseline))

    with open(eval_dir / version / 'annotations' / '{}_annotation_results.txt'.format(score_name)) as file:
        results = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    result_pages = list(map(lambda s: TruePage(systems=s), results))

    compare_results(result_pages, true_pages, score_name, version)


if __name__ == '__main__':
    version = 'current'
    scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l_Apprenti_Sorcier', 'Haydn_Symphony_104_London', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31', 'Schubert_Symphony_4', 'Van_Bree_Allegro']
    for score in scores:
        evaluate(score, version)

import json
import string

from score_analysis.barline_detector import System
from util.dirs import musicdata_dir

composer_name_map = {
    'bach': 'Bach, J.S.',
    'beethoven': 'Beethoven, L. van',
    'brahms': 'Brahms, J.',
    'bruckner': 'Bruckner, A.',
    'holst': 'Holst, G.',
    'mahler': 'Mahler, G.',
    'mozart': 'Mozart, W.A.',
    'tchaikovsky': 'Tchaikovsky, P.I.',
}


def generate_table_header(single_acc_only=True):
    layout_cols = '{llr}' if single_acc_only else '{llrr}'
    table = '\\begin{table}[]\n\\begin{tabular}' + layout_cols + '\n'
    accuracy_cols = '\\textbf{Avg. page accuracy}' if single_acc_only else '\\textbf{Dalitz avg. accuracy}\t&\t\\textbf{Meuleman avg. accuracy}'
    table += '\\textbf{Score}\t&\t\\textbf{Composer}\t&\t' + accuracy_cols + '\t\\\\\n'
    return table


def generate_table_footer():
    return '\\end{tabular}\n\\caption{}\n\\label{}\n\\end{table}'


def generate_table_row(dalitz_result, meuleman_result):
    score = dalitz_result['score']
    name_parts = [s.capitalize() for s in score.split('_')[1:]]
    if name_parts[0] == 'Symphony':
        name_parts = [name_parts[0], 'No.', name_parts[1]]
    name = ' '.join(name_parts)
    composer = composer_name_map[score.split('_')[0]]
    return '{}\t&\t{}\t&\t{}\t&\t{}\t\\\\\n'.format(name, composer, round(dalitz_result['avg_page_accuracy'], 4), round(meuleman_result['avg_page_accuracy'], 4))


def generate_table_results():
    table = generate_table_header()
    for score in sorted([score for score in musicdata_dir.iterdir() if score.is_dir()]):
        score_dir = musicdata_dir / score
        results = verify_musicdata_score(score_dir)
        name_parts = [s.capitalize() for s in score.name.split('_')[1:]]
        if name_parts[0] == 'Symphony':
            name_parts = [name_parts[0], 'No.', name_parts[1]]
        name = ' '.join(name_parts)
        composer = composer_name_map[score.name.split('_')[0]]
        table += '{}\t&\t{}\t&\t{}\t\\\\\n'.format(name, composer, round(results['avg_page_accuracy'], 4))
    table += generate_table_footer()
    return table


def generate_table_all_combined_results():
    table = generate_table_header(single_acc_only=False)
    for score in sorted([score for score in musicdata_dir.iterdir() if score.is_dir()]):
        score_dir = musicdata_dir / score
        dalitz_results = verify_musicdata_score(score_dir, 'Dalitz')
        meuleman_results = verify_musicdata_score(score_dir, 'Meuleman')
        table += generate_table_row(dalitz_results, meuleman_results)
    table += generate_table_footer()
    return table


def parse_category_errors(staff_finder):
    categories = []
    with open(musicdata_dir / ('staff_finder_errors_' + staff_finder + '.txt')) as f:
        file_str = f.read()
    for category_str in file_str.split('\n\n'):
        score_strs = category_str.split('\n')[1:]
        counts = sum([len(s.split(':')[1].split(',')) for s in score_strs])
        categories.append(counts)
    return categories


def generate_category_table():
    table = '\\begin{table}[]\n\\begin{tabular}{lrr}\n'
    table += '\\textbf{Category}\t&\t\\textbf{Dalitz count}\t&\t\\textbf{Meuleman count}\t\\\\\n'
    dalitz_errors = parse_category_errors('Dalitz')
    meuleman_errors = parse_category_errors('Meuleman')
    for i in range(len(dalitz_errors)):
        table += '{}\t&\t{}\t&\t{}\t\\\\\n'.format(string.ascii_uppercase[i], dalitz_errors[i], meuleman_errors[i])
    table += generate_table_footer()
    return table


def print_results(results):
    print('Score {}'.format(results['score']))
    print('Page\tSystem\tTrue\tFound\tAccuracy')
    for page in results['page_results']:
        if page['page_accuracy'] < 1:
            print('{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(page['page'], page['system'], page['true_barlines'], page['found_barlines'], page['page_accuracy']))
    print('Misaligned systems:')
    for misalign in results['misaligned_systems']:
        if misalign['misaligned_systems'] > 0:
            print('\t\tPage {}: {} misaligned'.format(misalign['page'], misalign['misaligned_systems']))
    print('Total accuracy {}, avg. page accuracy {}\n------------------------'.format(results['accuracy'], results['avg_page_accuracy']))


def combine_results(result_array, title=None):
    avg = sum([r['avg_page_accuracy'] for r in result_array]) / len(result_array)
    if title is not None:
        print(title)
    print('Total avg page accuracy {}'.format(avg))


def get_staffs_path_sort_key(path):
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
        staffs_paths = []
        for part in combine_parts:
            staffs_paths.extend(sorted([file for file in (part / 'barlines').iterdir()], key=get_staffs_path_sort_key))
    else:
        staffs_paths = sorted([file for file in (score_dir / 'barlines').iterdir()], key=get_staffs_path_sort_key)
    if len(staffs_paths) != len(annotations):
        raise AssertionError(
            '{}: Found {} pages, expected {}'.format(score_dir.parts[-1], len(staffs_paths),
                                                         len(annotations)))
    true_avg_count = found_avg_count = 0
    page_accuracy_sum = 0
    page_results = []
    misaligned_systems = []
    for i, path in enumerate(staffs_paths):
        with open(path) as f:
            systems = [System.from_json(json_data) for json_data in json.load(f)['systems']]
        misaligned_page_systems = []
        for system in systems:
            if len(system.barlines) == 1:
                misaligned_page_systems.append(system)
        misaligned_systems.append({'page': i + 1, 'misaligned_systems': len(misaligned_page_systems)})
        for system in misaligned_page_systems:
            systems.remove(system)
        if len(systems) != len(annotations[i]):
            raise AssertionError('Page {}: Found {} systems, expected {}'.format(i + i, len(systems), len(annotations[i])))
        for j, system in enumerate(systems):
            true_barlines = annotations[i][j][1] + 1
            true_avg_count += true_barlines
            found_barlines = len(system.barlines)
            found_avg_count += found_barlines
            page_accuracy = 1 - (abs(true_barlines - found_barlines) / true_barlines)
            page_accuracy_sum += page_accuracy
            page_results.append({'page': i + 1, 'system': j + 1, 'found_barlines': found_barlines, 'true_barlines': true_barlines, 'page_accuracy': page_accuracy})
    accuracy = 1 - (abs(true_avg_count - found_avg_count) / true_avg_count)
    page_accuracy_avg = page_accuracy_sum / sum([len(a) for a in annotations])
    results = {
        'score': score_dir.parts[-1],
        'accuracy': accuracy,
        'avg_page_accuracy': page_accuracy_avg,
        'page_results': page_results,
        'misaligned_systems': misaligned_systems,
    }
    return results


def main():
    # print(generate_table_results('Dalitz'))
    # print(generate_table_all_combined_results())
    # print(generate_category_table())
    for score in sorted([score for score in musicdata_dir.iterdir() if score.is_dir()]):
        print_results(verify_musicdata_score(score))


if __name__ == '__main__':
    main()

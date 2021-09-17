import json
import string
from pathlib import Path

musicdata_dir = Path(__file__).parents[2] / 'OMR-measure-segmenter-data/musicdata'

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


def generate_table_results(staff_finder):
    table = generate_table_header()
    for score in sorted([score for score in musicdata_dir.iterdir() if score.is_dir()]):
        score_dir = musicdata_dir / score
        results = verify_musicdata_score(score_dir, staff_finder)
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
    print('Score {} with {}'.format(results['score'], results['staff_finder']))
    print('Page\tFound\tTrue\tAccuracy')
    for page in results['page_results']:
        if page['page_accuracy'] < 1:
            print('{}\t\t{}\t\t{}\t\t{}'.format(page['page'], page['true_staffs'], page['found_staffs'], page['page_accuracy']))
    print('Total accuracy {}, avg. page accuracy {}\n------------------------'.format(results['accuracy'], results['avg_page_accuracy']))


def combine_results(result_array, title=None):
    avg = sum([r['avg_page_accuracy'] for r in result_array]) / len(result_array)
    if title is not None:
        print(title)
    print('Total avg page accuracy {}'.format(avg))


def find_staff_finder_differences(score_dir):
    dalitz_results = verify_musicdata_score(score_dir, 'Dalitz')
    meuleman_results = verify_musicdata_score(score_dir, 'Meuleman')
    print('Score {}'.format(score_dir.parts[-1]))
    print('Page\tTrue\tDalitz\tMeuleman')
    for i in range(len(dalitz_results['page_results'])):
        true_staffs = dalitz_results['page_results'][i]['true_staffs']
        meuleman_found = meuleman_results['page_results'][i]['found_staffs']
        dalitz_found = dalitz_results['page_results'][i]['found_staffs']
        if meuleman_found != dalitz_found:
            print('{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(i + 1, true_staffs, dalitz_found, meuleman_found,
                                                      '!!' if meuleman_found != true_staffs else ''))
    print('---------------------')


def find_meuleman_errors(score_dir, ignore_dalitz_errors=True):
    dalitz_results = verify_musicdata_score(score_dir, 'Dalitz')
    meuleman_results = verify_musicdata_score(score_dir, 'Meuleman')
    print('Score {}'.format(score_dir.parts[-1]))
    print('Page\tTrue\tDalitz\tMeuleman')
    for i in range(len(dalitz_results['page_results'])):
        true_staffs = dalitz_results['page_results'][i]['true_staffs']
        meuleman_found = meuleman_results['page_results'][i]['found_staffs']
        dalitz_found = dalitz_results['page_results'][i]['found_staffs']
        if meuleman_found != true_staffs and (not ignore_dalitz_errors or meuleman_found == dalitz_found):
            print('{}\t\t{}\t\t{}\t\t{}'.format(i + 1, true_staffs, dalitz_found, meuleman_found))
    print('---------------------')


def get_staffs_path_sort_key(path):
    return int(path.stem.split('_')[1])


def verify_musicdata_score(score_dir, staff_finder):
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
            staffs_paths.extend(sorted([file for file in (part / 'staffs' / staff_finder).iterdir()], key=get_staffs_path_sort_key))
    else:
        staffs_paths = sorted([file for file in (score_dir / 'staffs' / staff_finder).iterdir()], key=get_staffs_path_sort_key)
    if len(staffs_paths) != len(annotations):
        raise AssertionError(
            '{}, {}: Found {} pages, expected {}'.format(score_dir.parts[-1], staff_finder, len(staffs_paths),
                                                         len(annotations)))
    true_avg_count = found_avg_count = 0
    page_accuracy_sum = 0
    page_results = []
    for i, path in enumerate(staffs_paths):
        true_staffs = sum([x[0] for x in annotations[i]])
        true_avg_count += true_staffs
        with open(path) as f:
            staffs = json.load(f)
        found_staffs = len(staffs['staves'])
        found_avg_count += found_staffs
        page_accuracy = 1 - (abs(true_staffs - found_staffs) / true_staffs)
        page_accuracy_sum += page_accuracy
        page_results.append({'page': i + 1, 'found_staffs': found_staffs, 'true_staffs': true_staffs, 'page_accuracy': page_accuracy})
    accuracy = 1 - (abs(true_avg_count - found_avg_count) / true_avg_count)
    page_accuracy_avg = page_accuracy_sum / len(annotations)
    results = {
        'score': score_dir.parts[-1],
        'staff_finder': staff_finder,
        'accuracy': accuracy,
        'avg_page_accuracy': page_accuracy_avg,
        'page_results': page_results,
    }
    return results


def main():
    # print(generate_table_results('Dalitz'))
    # print(generate_table_all_combined_results())
    print(generate_category_table())


if __name__ == '__main__':
    main()

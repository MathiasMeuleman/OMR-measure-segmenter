from collections import namedtuple

from util.dirs import get_musicdata_scores

composer_name_map = {
    'bach': 'Bach, J.S.',
    'beethoven': 'Beethoven, L.\\ van',
    'brahms': 'Brahms, J.',
    'bruckner': 'Bruckner, A.',
    'holst': 'Holst, G.',
    'mahler': 'Mahler, G.',
    'mozart': 'Mozart, W.A.',
    'tchaikovsky': 'Tchaikovsky, P.I.',
}


TableColumn = namedtuple('TableColumn', ['title', 'accessor', 'align'])


def generate_table_header(columns):
    score_col_aligns = 'll'
    column_aligns = '{' + score_col_aligns + ''.join([c.align for c in columns]) + '}'
    table = '\\begin{table}[]\n\\begin{tabular}' + column_aligns + '\n'
    score_col_names = '\t&\t'.join(['\\textbf{' + n + '}' for n in ['Score', 'Composer']])
    column_names = '\t&\t'.join(['\\textbf{' + c.title + '}' for c in columns])
    table += score_col_names + '\t&\t' + column_names + '\t\\\\\n'
    return table


def generate_table_footer():
    return '\\end{tabular}\n\\caption{}\n\\label{}\n\\end{table}'


def generate_table(columns, data):
    table = generate_table_header(columns)
    for datapoint in data:
        score = datapoint['score']
        name_parts = [s.capitalize() for s in score.split('_')[1:]]
        if name_parts[0] == 'Symphony':
            name_parts = [name_parts[0], 'No.\\', name_parts[1]]
        if name_parts[0] == 'Brandenburg':
            name_parts = [name_parts[0], name_parts[1], 'No.\\', '5']
        name = ' '.join(name_parts)
        composer = composer_name_map[score.split('_')[0]]
        score_entries = '\t&\t'.join([name, composer])
        datapoint = next(d for d in data if d['score'] == score)
        column_entries = '\t&\t'.join([str(datapoint[c.accessor]) for c in columns])
        table += score_entries + '\t&\t' + column_entries + '\t\\\\\n'
    table += generate_table_footer()
    return table

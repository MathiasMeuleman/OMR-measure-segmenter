import json
from itertools import product
from os.path import dirname, join, realpath
import matplotlib.pyplot as plt

test_dir = join(dirname(dirname(dirname(realpath(__file__)))), 'OMR-measure-segmenter-data/stafffinder-testset')
excluded_param_keys = ['infile', 'nostaves', 'algorithm', 'staff_finder', 'deformation', 'error', 'metric', 'time']

deformations = {
    'no-deform': {},
    'rotation': {'angle': range(-18, 19)},
    'curvature': {'ampx': [x * 0.02 for x in range(1, 16)]},
    'typeset_emulation': {'n_gap': [10], 'n_shift': range(1, 11), 'p_gap': [0.5]},
    'staffline_interruptions': {'alpha': [x * 0.01 for x in range(1, 11)], 'n': [6], 'p': [0.5]},
    'staffline_thickness_variation': {'c': [0.8], 'min': [1], 'max': range(2, 11)},
    'staffline_y_variation': {'c': [0.8], 'maxdiff': range(2, 11)},
    # 'degrade_kanungo_parallel': {'eta': [0], 'k': [2], 'a0': [0.5, 1], 'a': [x * 0.25 for x in range(1, 7)], 'b0': [0.5, 1], 'b': [x * 0.25 for x in range(1, 7)]},
    # 'white_speckles_parallel': {'k': [2], 'n': [10], 'p': [x * 0.01 for x in range(1, 51)]},
}

# deformation_sets = {}
# for k, v in deformations.iteritems():
#     keys = list(v.keys())
#     cartesian = list(product(*list(v.values())))
#     sets = [dict(zip(keys, c)) for c in cartesian]
#     deformation_sets[k] = sets

algorithms = {
    'linetracking_height': {'linestyle': '-.'},
    'linetracking_chord': {'linestyle': '-', 'dashes': [8, 4, 8, 4, 2, 4]},
    'roach_tatem_original': {'linestyle': ''},
    'roach_tatem_improved': {'linestyle': '--'},
    'skeletonization': {'linestyle': '-'}
}
staff_finders = ['Dalitz', 'Dalitz_0', 'Meuleman']
colors = {'Dalitz': 'k', 'Dalitz_0': 'r', 'Meuleman': 'y'}


def inspect_results(results):
    unique_files = set([r['infile'] for r in results])
    for file in unique_files:
        print(file)
        selected_results = [r for r in results if r['infile'] == file]
        deformations = set([r['deformation'] for r in selected_results])
        for deformation in deformations:
            deformation_results = [r for r in selected_results if r['deformation'] == deformation]
            print('\t' + deformation + '  /  ' + str(list(set([r['staff_finder'] for r in deformation_results]))))
            param_keys = [k for k in deformation_results[0].keys() if k not in excluded_param_keys]
            param_string = []
            for key in param_keys:
                vals = [r[key] for r in deformation_results]
                if min(vals) < max(vals):
                    param_string.append('' + key + ': ' + str([min(vals), max(vals)]))
            print('\t\t' + ', '.join(param_string))


def inspect_results_file(filename):
    with open(join(test_dir, filename)) as file:
        results = json.loads(file.read())['results']
    inspect_results(results)


def extract_selected_from_results(sourcefile, targetfile):
    with open(join(test_dir, sourcefile)) as file:
        results = json.loads(file.read())['results']
    selected = [r for r in results if all([
        r['deformation'] in deformations and deformations[r['deformation']][k][0] <= round(r[k], 3) <= deformations[r['deformation']][k][-1]
        for k in [k for k in r.keys() if k not in excluded_param_keys]])
                ]
    inspect_results(selected)
    with open(join(test_dir, targetfile), 'w') as file:
        file.write(json.dumps(selected, indent=2, sort_keys=True))


def get_total_pixel_error(results):
    total_wrong_pixels = sum([r['error']['e1'] + r['error']['e2'] for r in results])
    total_black_pixels = sum([r['error']['a'] for r in results])
    return float(total_wrong_pixels) / float(total_black_pixels)


def get_total_segment_error(results):
    totals = [sum([r['error']['n' + str(i)] for r in results]) for i in range(1, 7)]
    return float(sum(totals) - totals[0]) / float(sum(totals))


def get_total_interrupt_error(results):
    totals = [sum([r['error']['n' + str(i)] for r in results]) for i in range(1, 4)]
    return float(min(totals[2], totals[0] + totals[1])) / float(totals[2])


def draw_graph(results, deformation, metric, param, fixed_params={}, yticks={}):
    selected = [r for r in results
                if r['deformation'] == deformation
                and r['metric'] == metric
                and all([r[k] == fixed_params[k] for k in fixed_params.keys()])]
    unique_param_vals = sorted(list(set([r[param] for r in selected])))
    removal_finder_pairs = [p for p in list(product([a for a in algorithms], ['Meuleman', 'Dalitz_0'])) if p[0] != 'roach_tatem_original']

    data = {param: unique_param_vals}
    plot_settings = []
    for (algorithm, staff_finder) in removal_finder_pairs:
        label = '' + algorithm + '_' + staff_finder
        color = colors[staff_finder]
        linestyle = color + algorithms[algorithm]['linestyle']
        setting = {'label': label, 'linestyle': linestyle}
        if 'dashes' in algorithm:
            setting['dashes'] = algorithm['dashes']
        plot_settings.append(setting)
        errors = []
        for val in unique_param_vals:
            group = [r for r in selected if r['algorithm'] == algorithm and r['staff_finder'] == staff_finder and r[param] == val]
            total_error = 0
            if metric == 'pixel':
                total_error = get_total_pixel_error(group)
            elif metric == 'segmentation':
                total_error = get_total_segment_error(group)
            elif metric == 'interruption':
                total_error = get_total_interrupt_error(group)
            errors.append(total_error)
        data[label] = errors
    plt.figure()
    for setting in plot_settings:
        line = plt.plot(param, setting['label'], setting['linestyle'], data=data)
        if 'dashes' in setting:
            line.set_dashes(setting['dashes'])
    plt.legend()
    plt.title(' '.join(deformation.split('_')).capitalize() + ', ' + str(fixed_params))
    if len(yticks.keys()) > 0:
        plt.yticks(**yticks)
    plt.show()


def combine(files, outfile):
    results = []
    for file in files:
        with open(join(test_dir, file)) as f:
            results.extend(json.loads(f.read())['results'])
    with open(join(test_dir, outfile), 'w') as f:
        f.write(json.dumps({'results': results}, indent=2, sort_keys=True))


graphs = [
    {'deformation': 'rotation', 'metric': 'segmentation', 'param': 'angle'},
    {'deformation': 'curvature', 'metric': 'segmentation', 'param': 'ampx'},
    {'deformation': 'typeset_emulation', 'metric': 'segmentation', 'param': 'n_shift', 'fixed_params': {'n_gap': 10}},
    {'deformation': 'staffline_thickness_variation', 'metric': 'segmentation', 'param': 'max', 'fixed_params': {'c': 0.8}},
    {'deformation': 'staffline_y_variation', 'metric': 'segmentation', 'param': 'maxdiff', 'fixed_params': {'c': 0.8}},
    {'deformation': 'staffline_interruptions', 'metric': 'pixel', 'param': 'alpha', 'fixed_params': {'n': 6}, 'yticks': {'ticks': [x / 20 for x in range(0, 9)]}},
    {'deformation': 'white_speckles_parallel', 'metric': 'segmentation', 'param': 'p', 'fixed_params': {'k': 2, 'n': 10}},
]


def draw_graphs(file='evaluation_results_selected.json'):
    with open(join(test_dir, file)) as f:
        results = json.loads(f.read())['results']
    for graph in graphs:
        draw_graph(results=results, **graph)


if __name__ == '__main__':
    # combine(['evaluation_results_Dalitz_0.json', 'evaluation_results_white_speckles_Dalitz_0.json'], 'evaluation_results_Dalitz_0_updated.json')
    # draw_graphs()

    with open(join(test_dir, 'evaluation_results_selected.json')) as f:
        results = json.loads(f.read())['results']
    selected = [r for r in results if r['deformation'] == 'rotation' and r['angle'] == -5]
    removal_finder_pairs = [p for p in list(product([a for a in algorithms], ['Meuleman', 'Dalitz', 'Dalitz_0'])) if p[0] != 'roach_tatem_original']
    for (algorithm, staff_finder) in removal_finder_pairs:
        group = [r for r in selected if r['algorithm'] == algorithm and r['staff_finder'] == staff_finder]
        for metric in ['pixel', 'segmentation', 'interruption']:
            total_error = 0
            metric_group = [r for r in group if r['metric'] == metric]
            if metric == 'pixel':
                total_error = get_total_pixel_error(metric_group)
            elif metric == 'segmentation':
                total_error = get_total_segment_error(metric_group)
            elif metric == 'interruption':
                total_error = get_total_interrupt_error(metric_group)
            print('\t'.join([algorithm, staff_finder, metric, str(total_error)]))

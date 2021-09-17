import json
import subprocess
from pathlib import Path
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from score_analysis.staff_verifyer import StaffVerifyer
import matplotlib.pyplot as plt

deformations = {
    'no-deform': {},
    # 'rotation': {'angle': range(-18, 19)},
    # 'curvature': {'ampx': [x * 0.02 for x in range(1, 16)]},
    # 'typeset_emulation': {'n_gap': [10], 'n_shift': range(1, 11), 'p_gap': [0.5]},
    # 'staffline_interruptions': {'alpha': [x * 0.01 for x in range(1, 11)], 'n': [6], 'p': [0.5]},
    # 'staffline_thickness_variation': {'c': [0.8], 'min': [1], 'max': range(2, 11)},
    # 'staffline_y_variation': {'c': [0.8], 'maxdiff': range(2, 11)},
    # 'degrade_kanungo_parallel': {'eta': [0], 'k': [2], 'a0': [0.5, 1], 'a': [x * 0.25 for x in range(1, 7)], 'b0': [0.5, 1], 'b': [x * 0.25 for x in range(1, 7)]},
    # 'white_speckles_parallel': {'k': [2], 'n': [10], 'p': [x * 0.01 for x in range(1, 51)]},
}
colors = {'Dalitz': 'k', 'Dalitz_0': 'r', 'Meuleman': 'y'}

deformation_sets = {}
for k, v in deformations.items():
    keys = list(v.keys())
    cartesian = list(product(*list(v.values())))
    sets = [dict(zip(keys, c)) for c in cartesian]
    deformation_sets[k] = sets


def get_filepaths():
    test_dir = Path(__file__).parents[2] / 'OMR-measure-segmenter-data/stafffinder-testset'
    test_cats = ['historic', 'modern', 'tablature']
    input_images = []
    for cat in test_cats:
        png_files = list((test_dir / cat).glob('*.png'))
        input_images.extend([f for f in png_files if not 'nostaff' in f.name])
    return input_images


def get_num_lines_arg(staff_finder_name, infile):
    if staff_finder_name == 'Meuleman' or staff_finder_name == 'Dalitz_0':
        return 0
    if 'tablature' in infile.parts:
        return 6
    if 'historic' in infile.parts and infile.stem == 'gregorian':
        return 4
    return 5


def draw_no_deform_staffs():
    directory = Path(__file__).parents[2] / 'OMR-measure-segmenter-data/stafffinder-testset/tablature'
    staffs_path = directory.parent / 'testset_staffs_no-deform'
    for staff_finder in ['Dalitz', 'Dalitz_0', 'Meuleman']:
        overlay_path = directory.parent / ('testset_no-deform_overlays_' + staff_finder)
        overlay_path.mkdir(exist_ok=True)
        verifyer = StaffVerifyer(directory=directory, staffs_path=staffs_path, overlay_path=overlay_path)
        for file in [f for f in list(directory.glob('*.png')) if 'nostaff' not in f.name]:
            print(file.stem + '_' + staff_finder + '*.json')
            staff = next(staffs_path.glob(file.stem + '_' + staff_finder + '*.json'))
            verifyer.overlay_page_staffs(0, file.name, staff.name, file.stem + '.png')


def run_py2_deform_and_detect_staffs(skip_deform, image_path, nostaff_path, method, params, output_path):
    detect_staffs_path = (Path(__file__).parents[1] / 'score_analysis/py2_deform_and_detect_staffs.py').absolute()
    param_args = ','.join([str(k) + '=' + str(params[k]) for k in params])
    subprocess.run(['/usr/bin/python', detect_staffs_path, image_path.absolute(), nostaff_path.absolute(), output_path.absolute(), method, '--params', param_args, '--skip-deform', str(skip_deform)])


def detect_testset_staffs():
    input_files = get_filepaths()
    output_path = Path(__file__).parents[2] / 'OMR-measure-segmenter-data/stafffinder-testset/testset_staffs_no-deform'
    arg_tuples = []
    for input_file in tqdm(input_files):
        nostaff_input_file = input_file.parent / (input_file.stem + '-nostaff.png')
        for method in deformation_sets.keys():
            if method == 'no-deform':
                arg_tuples.append((False, input_file, nostaff_input_file, method, {}, output_path))
            else:
                for params in deformation_sets[method]:
                    arg_tuples.append((True, input_file, nostaff_input_file, method, params, output_path))
    pool = Pool(10)
    pool.starmap(run_py2_deform_and_detect_staffs, arg_tuples)
    print('\n\n=============\nDONE!\n================')


def is_numerical(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def parse_filename(filepath):
    methods = [g['method'] for g in verification_group]
    parts = '.'.join(str(filepath.absolute()).split('/')[-1].split('.')[:-1]).split('_')
    staff_finder = parts[1]
    if parts[2] == '0':
        staff_finder = 'Dalitz_0'
        del parts[2]
    k = 3
    while '_'.join(parts[2:k]) not in methods:
        k += 1
    method = '_'.join(parts[2:k])
    param_parts = parts[k:len(parts)]
    params = {}
    i = j = 0
    while i < len(param_parts):
        while not is_numerical(param_parts[j]):
            j += 1
        params['_'.join(param_parts[i:j])] = float(param_parts[j])
        i = j = j + 1
    for param in params:
        if int(params[param]) == params[param]:
            params[param] = int(params[param])
    return {'file': filepath, 'image': parts[0], 'staff_finder': staff_finder, 'method': method, 'params': params}


def verify_group(files, staff_annotations, method, param, fixed_params={}, yticks={}):
    true_group_count = sum([staff_annotations[a]['staffs'] for a in staff_annotations])
    true_staffline_count = sum([staff_annotations[a]['staffs'] * staff_annotations[a]['stafflines'] for a in staff_annotations])
    selected = [f for f in files
                if f['method'] == method
                and all([f['params'][k] == fixed_params[k] for k in fixed_params.keys()])]
    unique_param_vals = sorted(list(set([f['params'][param] for f in selected])))
    staff_data = {param: unique_param_vals}
    staffline_data = {param: unique_param_vals}
    plot_settings = []
    for staff_finder in ['Dalitz', 'Dalitz_0', 'Meuleman']:
        color = colors[staff_finder]
        setting = {'label': staff_finder, 'linestyle': color}
        plot_settings.append(setting)
        errors = []
        group_errors = []
        for val in unique_param_vals:
            selected_files = [f for f in selected if f['staff_finder'] == staff_finder and f['params'][param] == val]
            group_count = 0
            staffline_count = 0
            for f in selected_files:
                with open(f['file']) as file:
                    staves = json.loads(file.read())['staves']
                    group_count += len(staves)
                    staffline_count += sum([len(stave['lines']) for stave in staves])
            total_error = abs(true_staffline_count - staffline_count) / true_staffline_count
            group_errors.append(group_count - true_group_count)
            errors.append(total_error)
        staff_data[staff_finder] = group_errors
        staffline_data[staff_finder] = errors

    plt.figure()
    for setting in plot_settings:
        line = plt.plot(param, setting['label'], setting['linestyle'], data=staffline_data)
        if 'dashes' in setting:
            line.set_dashes(setting['dashes'])
    plt.legend()
    plt.title('Stafflines: ' + ' '.join(method.split('_')).capitalize() + ', ' + str(fixed_params))
    if len(yticks.keys()) > 0:
        plt.yticks(**yticks)
    plt.show()

    plt.figure()
    for setting in plot_settings:
        line = plt.plot(param, setting['label'], setting['linestyle'], data=staff_data)
        if 'dashes' in setting:
            line.set_dashes(setting['dashes'])
    plt.legend()
    plt.title('Staffs: ' + ' '.join(method.split('_')).capitalize() + ', ' + str(fixed_params))


verification_group = [
    {'method': 'rotation', 'param': 'angle'},
    {'method': 'curvature', 'param': 'ampx'},
    {'method': 'typeset_emulation', 'param': 'n_shift', 'fixed_params': {'n_gap': 10}},
    {'method': 'staffline_thickness_variation', 'param': 'max', 'fixed_params': {'c': 0.8}},
    {'method': 'staffline_y_variation', 'param': 'maxdiff', 'fixed_params': {'c': 0.8}},
    {'method': 'staffline_interruptions', 'param': 'alpha', 'fixed_params': {'n': 6}},
    {'method': 'white_speckles_parallel', 'param': 'p', 'fixed_params': {'k': 2, 'n': 10}},
]


def verify():
    testset_staffs_dir = Path(__file__).parents[2] / 'OMR-measure-segmenter-data/stafffinder-testset/testset_staffs'
    files = list(map(parse_filename, testset_staffs_dir.glob('*.json')))
    with open(testset_staffs_dir.parent / 'staff_annotations.txt') as file:
        staff_annotations_raw = file.readlines()
    del staff_annotations_raw[0]
    staff_annotations = {}
    for anno in staff_annotations_raw:
        parts = anno.split(' ')
        name = parts[0].split('/')[1].split('.')[0]
        staff_annotations[name] = {'stafflines': int(parts[1]), 'staffs': int(parts[2])}
    for graph in verification_group:
        verify_group(files, staff_annotations, **graph)


if __name__ == '__main__':
    # verify()
    draw_no_deform_staffs()

    # detect_testset_staffs()
    # print(parse_filename('bach_Dalitz_0_staffline_interruptions_alpha_0.05_p_gap_0.5_n_6.json'))

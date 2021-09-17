"""
IMPORTANT!! Only runs on Python 2.
"""
import json
from os.path import basename, join
from argparse import ArgumentParser
from gamera.core import *
from gamera.toolkits.musicstaves.stafffinder_dalitz import StaffFinder_dalitz
from gamera.toolkits.musicstaves.stafffinder_meuleman import StaffFinder_meuleman


def get_num_lines_arg(staff_finder_name, infile):
    if staff_finder_name == 'Meuleman' or staff_finder_name == 'Dalitz_0':
        return 0
    if 'tablature' in infile:
        return 6
    if 'historic' in infile and 'gregorian' in infile:
        return 4
    return 5


def deform_image(image, image_staff_only, method, params):
    if method == 'no-deform':
        return image.rotation(im_staffonly=image_staff_only, angle=0)
    args = params.copy()
    args['im_staffonly'] = image_staff_only
    return getattr(image, method)(**args)


def detect_staffs(skip_deform, image_path, nostaff_path, method, params, output_dir):
    init_gamera()
    image = load_image(image_path)
    nostaves = load_image(nostaff_path)
    if image.data.pixel_type != ONEBIT:
        image = image.to_onebit()
    if nostaves.data.pixel_type != ONEBIT:
        nostaves = nostaves.to_onebit()
    image_staff_only = image.subtract_images(nostaves)
    print(str({'image': basename(image_path), 'method': method, 'params': params}))
    if skip_deform:
        print('Skipping deformation')
        work_img = image
    else:
        [deformed_img, _, _] = deform_image(image, image_staff_only, method, params)
        work_img = deformed_img

    for staff_finder_name in ['Dalitz', 'Dalitz_0', 'Meuleman']:
        print('\t' + staff_finder_name + '...')
        staff_finder = StaffFinder_meuleman(work_img) if staff_finder_name == 'Meuleman' else StaffFinder_dalitz(work_img)
        staff_finder.find_staves(num_lines=get_num_lines_arg(staff_finder_name, image_path), debug=0)

        staves = staff_finder.get_average()
        staff_points = [{'lines': [[[line.left_x, line.average_y], [line.right_x, line.average_y]] for line in staff]} for staff in staves]
        out_filename = basename(image_path).split('.')[0] + '_' + staff_finder_name + '_' + method + '_' + '_'.join(
            [str(k) + '_' + str(params[k]) for k in params]) + '.json'
        output_path = join(output_dir, out_filename)
        with open(output_path, 'w') as file:
            file.write(json.dumps({'staves': staff_points}, indent=2, sort_keys=True))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('nostaff_image_path')
    parser.add_argument('output_path')
    parser.add_argument('method')
    parser.add_argument('--skip-deform', type=bool)
    parser.add_argument('--params')
    args = parser.parse_args()
    if not ',' in args.params:
        params = {}
    else:
        params = dict([[v.split('=')[0], float(v.split('=')[1])] for v in args.params.split(',')])
    for param in params:
        if int(params[param]) == params[param]:
            params[param] = int(params[param])
    detect_staffs(args.skip_deform, args.image_path, args.nostaff_image_path, args.method, params, args.output_path)

"""
IMPORTANT!! Only runs on Python 2.
"""
import json
import sys
from gamera.core import *
from staff_finder_py2 import stafffinder_meuleman


def detect_staffs(image_path, output_path):
    init_gamera()
    image = load_image(image_path)
    image.to_onebit()

    sf = stafffinder_meuleman.StaffFinder_meuleman(image)
    sf.find_staves(num_lines=0, debug=0)

    staves = sf.get_average()
    staff_points = [{'lines': [[[line.left_x, line.average_y], [line.right_x, line.average_y]] for line in staff]} for staff in staves]
    with open(output_path, 'w') as file:
        file.write(json.dumps({'staves': staff_points}, indent=2, sort_keys=True))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise RuntimeError('Expected two arguments: `python py2_detect_staffs.py <image_path> <output_path>')
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    detect_staffs(image_path, output_path)

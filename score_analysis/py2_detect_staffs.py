"""
IMPORTANT!! Only runs on Python 2.
"""
import json
import sys
from gamera.core import *
from gamera.toolkits.musicstaves.stafffinder_dalitz import StaffFinder_dalitz
from gamera.toolkits.musicstaves.stafffinder_meuleman import StaffFinder_meuleman


def detect_staffs(image_path, output_path, staff_finder_name='Meuleman'):
    init_gamera()
    image = load_image(image_path)
    image.to_onebit()

    staff_finder = StaffFinder_meuleman(image) if staff_finder_name == 'Meuleman' else StaffFinder_dalitz(image)
    print('Detecting staffs in ' + image_path.split('/')[-1] + ' with StaffFinder ' + staff_finder.__class__.__name__)
    numlines = 5 if staff_finder_name == 'Dalitz' else 0
    staff_finder.find_staves(num_lines=numlines, debug=0)

    staves = staff_finder.get_average()
    staff_points = [{'lines': [[[line.left_x, line.average_y], [line.right_x, line.average_y]] for line in staff]} for staff in staves]
    with open(output_path, 'w') as file:
        file.write(json.dumps({'staves': staff_points}, indent=2, sort_keys=True))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError('Expected two arguments: `python py2_detect_staffs.py <image_path> <output_path>')
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    staff_finder_name = sys.argv[3] if len(sys.argv) > 3 else None
    detect_staffs(image_path, output_path, staff_finder_name)

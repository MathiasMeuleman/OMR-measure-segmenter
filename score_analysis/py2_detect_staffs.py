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

    staff_finder_average = staff_finder.get_average()
    staffs = []
    for staff in staff_finder_average:
        staff_obj = Staff()
        for line in staff:
            staffline = StaffLine(line.left_x, line.right_x, line.average_y)
            staff_obj.add_staffline(staffline)
        staffs.append(staff_obj)
    staff_collection = {'staffs': [staff.to_json() for staff in staffs]}
    with open(output_path, 'w') as file:
        json.dump(staff_collection, file, indent=2, sort_keys=True)


class StaffLine:
    """
    Data class that represents a staffline.
    """
    def __init__(self, start, end, y):
        self.start = start
        self.end = end
        self.y = y

    def to_json(self):
        return {'start': self.start, 'end': self.end, 'y': self.y}


class Staff:

    def __init__(self):
        self.stafflines = []
        self.start = 0
        self.end = 0
        self.top = 0
        self.bottom = 0

    def add_staffline(self, staffline):
        first = len(self.stafflines) == 0
        self.stafflines.append(staffline)
        if first or staffline.start < self.start:
            self.start = staffline.start
        if first or staffline.end > self.end:
            self.end = staffline.end
        if first or staffline.y < self.top:
            self.top = staffline.y
        if first or staffline.y > self.bottom:
            self.bottom = staffline.y

    def to_json(self):
        return {
            'stafflines': [staffline.to_json() for staffline in self.stafflines],
            'top': self.top,
            'bottom': self.bottom,
            'start': self.start,
            'end': self.end,
        }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError('Expected two arguments: `python py2_detect_staffs.py <image_path> <output_path>')
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    staff_finder_name = sys.argv[3] if len(sys.argv) > 3 else None
    detect_staffs(image_path, output_path, staff_finder_name)

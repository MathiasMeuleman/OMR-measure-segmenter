import json

from PIL import Image
from tqdm import tqdm

from score_analysis.score_image import ScoreImage
from score_analysis.stafffinder_meuleman import StaffFinder_meuleman
from util.dirs import data_dir
from util.score_draw import ScoreDraw


class StaffDetector:

    def __init__(self, image, output_path=None):
        self.image = image if isinstance(image, ScoreImage) else ScoreImage(image)
        self.output_path = output_path

    def detect_staffs(self):
        staff_finder = StaffFinder_meuleman(self.image)
        staff_finder.find_staves(debug=0)
        staff_finder_average = staff_finder.get_average()
        staffs = []
        for staff in staff_finder_average:
            staff_obj = Staff()
            for line in staff:
                staffline = StaffLine(line.left_x, line.right_x, line.average_y)
                staff_obj.add_staffline(staffline)
            staffs.append(staff_obj)
        if self.output_path is not None:
            with open(self.output_path, 'w') as f:
                f.write(json.dumps({'staffs': [staff.to_json() for staff in staffs]}, sort_keys=True, indent=2))
        return staffs


class StaffLine:
    """
    Data class that represents a staffline.
    """
    def __init__(self, start, end, y):
        self.start = start
        self.end = end
        self.y = y

    @staticmethod
    def from_json(json_data):
        staffline = StaffLine(json_data['start'], json_data['end'], json_data['y'])
        return staffline

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

    @staticmethod
    def from_json(json_data):
        staff = Staff()
        staff.top = json_data['top']
        staff.bottom = json_data['bottom']
        staff.start = json_data['start']
        staff.end = json_data['end']
        staff.stafflines = [StaffLine.from_json(staffline) for staffline in json_data['stafflines']]
        return staff

    def to_json(self):
        return {
            'stafflines': [staffline.to_json() for staffline in self.stafflines],
            'top': self.top,
            'bottom': self.bottom,
            'start': self.start,
            'end': self.end,
        }


if __name__ == '__main__':
    staff_path = data_dir / 'sample' / 'staffs'
    staff_path.mkdir(parents=True, exist_ok=True)
    overlay_path = data_dir / 'sample' / 'staff_overlays' / 'pages'
    overlay_path.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm((data_dir / 'sample' / 'pages').iterdir()):
        image = Image.open(image_path)
        staffs = StaffDetector(image, output_path=staff_path / (image_path.stem + '.json')).detect_staffs()
        score_draw = ScoreDraw(image)
        staff_image = score_draw.draw_staffs(staffs)
        staff_image.save(overlay_path / image_path.name)

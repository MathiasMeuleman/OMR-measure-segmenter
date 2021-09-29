import json
import subprocess
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from score_analysis.staff_detector import Staff
from util.dirs import data_dir
from util.score_draw import ScoreDraw


class StaffDetectorLegacy:

    def __init__(self, image_path, staff_finder, output_path=None):
        self.image_path = Path(image_path)
        self.staff_finder = staff_finder
        self.output_path = output_path

    def run_py2_detect_staffs(self, image_path, output_path):
        detect_staffs_path = (Path(__file__).parent / 'py2_detect_staffs.py').absolute()
        subprocess.run(['/usr/bin/python', detect_staffs_path, image_path.absolute(), output_path.absolute(), self.staff_finder])

    def detect_staffs(self):
        output_path = self.image_path.parents[0] / '.tmp-{}.json'.format(self.image_path.stem) if self.output_path is None else self.output_path
        self.run_py2_detect_staffs(self.image_path, output_path)
        with open(output_path) as f:
            staff_collection = json.load(f)
            staffs = [Staff.from_json(staff) for staff in staff_collection['staffs']]
        if self.output_path is None:
            output_path.unlink()
        return staffs


if __name__ == '__main__':
    staff_path = data_dir / 'sample' / 'legacy' / 'staffs'
    staff_path.mkdir(parents=True, exist_ok=True)
    overlay_path = data_dir / 'sample' / 'legacy' / 'staff_overlays' / 'pages'
    overlay_path.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm((data_dir / 'sample' / 'pages').iterdir()):
        image = Image.open(image_path)
        staffs = StaffDetectorLegacy(image_path, staff_finder='Dalitz',
                                     output_path=staff_path / (image_path.stem + '.json')).detect_staffs()
        score_draw = ScoreDraw(image)
        staff_image = score_draw.draw_staffs(staffs)
        staff_image.save(overlay_path / image_path.name)

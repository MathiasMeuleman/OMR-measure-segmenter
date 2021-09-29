import json

from PIL import Image, ImageColor, ImageDraw
from tqdm import tqdm

from score_analysis.score_image import ScoreImage
from score_analysis.stafffinder_meuleman import StaffFinder_meuleman
from util.dirs import data_dir


class StaffDetector:

    def __init__(self, image, output_path=None):
        self.image = image if isinstance(image, ScoreImage) else ScoreImage(image)
        self.output_path = output_path

    def detect_staffs(self):
        staff_finder = StaffFinder_meuleman(self.image)
        staff_finder.find_staves(debug=0)
        staves = staff_finder.get_average()
        staffs = {'staves': [
            {'lines': [[[line.left_x, line.average_y], [line.right_x, line.average_y]] for line in staff]}
            for staff in staves]}
        if self.output_path is not None:
            with open(self.output_path, 'w') as f:
                f.write(json.dumps(staffs, sort_keys=True, indent=2))
        return staffs


if __name__ == '__main__':
    staff_path = data_dir / 'sample' / 'staffs'
    staff_path.mkdir(parents=True, exist_ok=True)
    overlay_path = data_dir / 'sample' / 'staff_overlays' / 'pages'
    overlay_path.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm((data_dir / 'sample' / 'pages').iterdir()):
        image = Image.open(image_path)
        staffs = StaffDetector(image, output_path=staff_path / (image_path.stem + '.json')).detect_staffs()
        staff_image = image.convert('RGB')
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for i, staff in enumerate(staffs['staves']):
            color = ImageColor.getrgb(colors[i % len(colors)])
            for line in staff['lines']:
                draw = ImageDraw.Draw(staff_image, mode='RGBA')
                draw.line([tuple(point) for point in line], fill=color, width=5)
                del draw
        staff_image.save(overlay_path / image_path.name)

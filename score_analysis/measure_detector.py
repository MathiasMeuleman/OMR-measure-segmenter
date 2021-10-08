import json
from itertools import product
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from score_analysis.barline_detector import BarlineDetector, System
from score_analysis.score_image import ScoreImage
from score_analysis.staff_detector import StaffDetector, Staff
from score_analysis.staff_detector_legacy import StaffDetectorLegacy
from util.dirs import data_dir
from util.score_draw import ScoreDraw


class MeasureDetector:

    def __init__(self, image_path, staff_path=None, barline_path=None, output_path=None, legacy=False, staff_finder=None):
        if legacy and staff_finder is None:
            raise ValueError('Argument "staff_finder" is required when "legacy" is set to "True".')
        if legacy and staff_finder not in ['Dalitz', 'Meuleman']:
            raise ValueError('Argument "staff_finder" must be one of "Dalitz" or "Meuleman".')
        self.image_path = Path(image_path)
        self.staff_path = Path(staff_path) if staff_path is not None else None
        self.barline_path = Path(barline_path) if barline_path is not None else None
        self.output_path = Path(output_path) if output_path is not None else None
        self.legacy = legacy
        self.staff_finder = staff_finder
        self.staffs = None
        self.systems = None
        self.measures = None

    class Bar:
        """Temporary data class for a Bar, the segment in between two Barlines"""
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def load_staffs(self):
        if self.staff_path is not None and self.staff_path.exists():
            with open(self.staff_path) as f:
                staff_collection = json.load(f)
                self.staffs = [Staff.from_json(staff) for staff in staff_collection['staffs']]
        else:
            raise FileNotFoundError('Could not load staffs at path {}'.format(self.staff_path))

    def load_systems(self):
        if self.barline_path is not None and self.barline_path.exists():
            with open(self.barline_path) as f:
                system_collection = json.load(f)
                self.systems = [System.from_json(system) for system in system_collection['systems']]
        else:
            raise FileNotFoundError('Could not load barlines at path {}'.format(self.barline_path))

    def load(self):
        self.load_staffs()
        self.load_systems()

    def detect_measures(self, force_staffs=False, force_barlines=False):
        score_image = ScoreImage(Image.open(self.image_path))
        if not force_staffs and self.staff_path is not None and self.staff_path.exists():
            self.load_staffs()
        else:
            if self.legacy:
                staff_detector = StaffDetectorLegacy(self.image_path, self.staff_finder, output_path=self.staff_path)
            else:
                staff_detector = StaffDetector(score_image, output_path=self.staff_path)
            self.staffs = staff_detector.detect_staffs()

        if not force_barlines and self.barline_path is not None and self.barline_path.exists():
            self.load_systems()
        else:
            barline_detector = BarlineDetector(score_image, output_path=self.barline_path)
            self.systems = barline_detector.detect_barlines()

        measures = []
        staff_idx = 0
        for i, system in enumerate(self.systems):
            bars = []
            for j in range(len(system.barlines) - 1):
                bars.append(self.Bar(system.barlines[j].get_average(), system.barlines[j + 1].get_average()))
            system_staffs = []
            while staff_idx < len(self.staffs) and self.staffs[staff_idx].stafflines[0].y < system.max_y:
                system_staffs.append(self.staffs[staff_idx])
                staff_idx += 1
            for (bar, staff) in product(bars, system_staffs):
                measures.append(Measure(bar.start, staff.top, bar.end, staff.bottom, i))

        if self.output_path is not None:
            with open(self.output_path, 'w') as f:
                json.dump({'measures': [measure.to_json() for measure in measures]}, f, indent=2, sort_keys=True)
        self.measures = measures


class Measure:

    def __init__(self, start, top, end, bottom, system):
        self.start = start
        self.top = top
        self.end = end
        self.bottom = bottom
        self.system = system

    @staticmethod
    def from_json(json_data):
        return Measure(json_data['start'], json_data['top'], json_data['end'], json_data['bottom'], json_data['system'])

    def to_json(self):
        return {'start': self.start, 'top': self.top, 'end': self.end, 'bottom': self.bottom, 'system': self.system}


if __name__ == '__main__':
    staff_path = data_dir / 'sample' / 'staffs'
    barline_path = data_dir / 'sample' / 'barlines'
    output_path = data_dir / 'sample' / 'measures'
    output_path.mkdir(exist_ok=True, parents=True)
    overlay_path = data_dir / 'sample' / 'measure_overlays'
    overlay_path.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm((data_dir / 'sample' / 'pages').iterdir()):
        json_name = image_path.stem + '.json'
        measure_detector = MeasureDetector(image_path, staff_path=staff_path / json_name,
                                           barline_path=barline_path / json_name, output_path=output_path / json_name)
        measure_detector.detect_measures()
        score_draw = ScoreDraw(Image.open(image_path))
        image = score_draw.draw_measures(measure_detector.measures)
        image.save(overlay_path / image_path.name)

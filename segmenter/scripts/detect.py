import json
from collections import namedtuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from tqdm import tqdm

from segmenter.measure_detector import MeasureDetector
from util.dirs import eval_dir, data_dir
from util.files import get_sorted_page_paths

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def construct_page_overlay(detector):
    img = Image.fromarray(detector.rotated).convert('RGB')
    max_height = 950
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, system in enumerate(detector.page.systems):
        color = colors[i % len(colors)]
        color_rgb = ImageColor.getrgb(color)
        color_rgba = (color_rgb[0], color_rgb[1], color_rgb[2], 96)
        for measure in system.measures:
            draw = ImageDraw.Draw(img, mode='RGBA')
            draw.rectangle(((measure.ulx, measure.uly), (measure.lrx, measure.lry)), outline=color_rgb, fill=color_rgba, width=5)
            del draw
    scale = max_height / img.size[1]
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    return img


def draw_page(detector):
    img = construct_page_overlay(detector)
    img.show()


def save_annotations(detectors, score_name, version):
    for detector in detectors:
        page = detector.page
        systems = []
        for system in page.systems:
            staffs = list(map(lambda staff: staff._asdict(), system.staffs))
            system_measures = list(map(lambda system_measure: system_measure._asdict(), system.system_measures))
            measures = list(map(lambda measure: measure._asdict(), system.measures))
            system_bb = {k: system._asdict()[k] for k in ('ulx', 'uly', 'lrx', 'lry')}
            systems.append({'staffs': staffs, 'system_measures': system_measures, 'measures': measures, **system_bb})
        page_dict = {'height': page.height, 'width': page.width, 'rotation': page.rotation, 'systems': systems}
        folder_path = eval_dir / version / 'annotations' / score_name
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / (page.name + '.json')
        with open(file_path, 'w') as f:
            f.write(json.dumps(page_dict, cls=NpEncoder, indent=4))


def save_pages(detectors, score_name, version):
    folder_path = eval_dir / version / 'visualized'
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / (score_name + '_visualized.pdf')
    im1 = construct_page_overlay(detectors[0])
    im_list = []
    for i in range(1, len(detectors)):
        img = construct_page_overlay(detectors[i])
        im_list.append(img)
    im1.save(file_path, 'PDF', resolution=100.0, save_all=True, append_images=im_list)


def detect(score_path, version, sys_method='lines', mode='run'):
    score_name = score_path.split('/')[0]
    page_path = data_dir / score_path
    paths = get_sorted_page_paths(page_path)
    detectors = []
    for i, path in tqdm(enumerate(paths)):
        print(path)
        detector = MeasureDetector(path).detect(plot=(mode == 'debug'), sys_method=sys_method)
        if mode == 'debug':
            draw_page(detector)
        else:
            detectors.append(detector)
    if mode == 'run':
        save_annotations(detectors, score_name, version)
        save_pages(detectors, score_name, version)


def detect_dataset(version, mode='run'):
    if mode == 'debug':
        scores = ['debug']
    else:
        scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l_Apprenti_Sorcier', 'Haydn_Symphony_104_London', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31', 'Schubert_Symphony_4', 'Van_Bree_Allegro']
    for score in scores:
        detect(score + '/ppm-300', version, mode=mode)


def detect_measurebb(version):
    scores = [p.stem for p in (data_dir / 'MeasureBoundingBoxAnnotations_v2').glob('*') if p.stem != 'coco']
    for score in scores:
        detect('MeasureBoundingBoxAnnotations_v2/' + score + '/img', version, mode='debug')


if __name__ == '__main__':
    debug = False
    version = 'test_json'
    dataset = 'measurebb'
    if dataset == 'dataset':
        detect_dataset(version, mode=('debug' if debug is True else 'run'))
    elif dataset == 'measurebb':
        detect_measurebb(version)

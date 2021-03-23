from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from segmenter.dirs import data_dir, eval_dir, tmp_dir
from segmenter.measure_detector import MeasureDetector
from util.files import get_sorted_page_paths

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


def construct_page_overlay(detector):
    img = Image.fromarray(detector.rotated).convert('RGB')
    max_height = 950
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, system in enumerate(detector.page.systems):
        for measure in system.measures:
            for staff in measure.staffs:
                draw = ImageDraw.Draw(img)
                draw.rectangle(((staff.ulx, staff.uly), (staff.lrx, staff.lry)), outline=colors[i % len(colors)], fill=None, width=5)
                del draw
    scale = max_height / img.size[1]
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    return img


def draw_page(detector):
    img = construct_page_overlay(detector)
    img.show()


def save_annotations(detectors, score_name, version):
    annotations = ''
    for detector in detectors:
        page = detector.page
        system_annotations = list(map(lambda system: str(len(system.staff_boundaries)) + ',' + str(len(system.measures)), page.systems))
        annotations += ' '.join(system_annotations) + '\n'
    file_path = Path(eval_dir, version, 'annotations', '{}_annotation_results.txt'.format(score_name)).resolve()
    with open(file_path, 'w') as f:
        f.write(annotations)


def save_pages(detectors, score_name, version):
    pdf_filename = Path(eval_dir, version, 'visualized', '{}_visualized.pdf'.format(score_name)).resolve()
    im1 = construct_page_overlay(detectors[0])
    im_list = []
    for i in range(1, len(detectors)):
        img = construct_page_overlay(detectors[i])
        im_list.append(img)
    im1.save(pdf_filename, 'PDF', resolution=100.0, save_all=True, append_images=im_list)


def detect(score_name, sys_method='lines', measure_method='region', mode='run'):
    if mode == 'debug':
        page_path = Path(tmp_dir, 'single').resolve()
    else:
        page_path = Path(data_dir, score_name, 'ppm-300').resolve()
    paths = get_sorted_page_paths(page_path)
    detectors = []
    for i, path in tqdm(enumerate(paths)):
        print(path)
        detector = MeasureDetector(path).detect(plot=(mode == 'debug'), sys_method=sys_method, measure_method=measure_method)
        if mode == 'debug':
            draw_page(detector)
        detectors.append(detector)
    if mode == 'run':
        save_annotations(detectors, score_name, version)
        save_pages(detectors, score_name, version)


if __name__ == '__main__':
    debug = False
    version = 'current'
    if debug:
        scores = ['debug']
    else:
        scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l_Apprenti_Sorcier', 'Haydn_Symphony_104_London', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31', 'Schubert_Symphony_4', 'Van_Bree_Allegro']
        # scores = ['Beethoven_Sextet', 'Van_Bree_Allegro']
    for score in scores:
        detect(score, sys_method='lines', mode=('debug' if debug is True else 'run'))

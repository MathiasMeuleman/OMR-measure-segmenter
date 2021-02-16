from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from segmenter.old.image_util import data_dir, tmp_dir
from segmenter.measure_detector import MeasureDetector
from util.files import get_sorted_page_paths

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


def construct_page_overlay(detector):
    img = Image.fromarray(detector.original)
    max_height = 950
    for system in detector.page.systems:
        for measure in system.measures:
            # draw = ImageDraw.Draw(img)
            # draw.rectangle(((measure.ulx, measure.uly), (measure.lrx, measure.lry)), outline='blue', fill=None,
            #                width=10)
            # del draw
            for staff in measure.staffs:
                draw = ImageDraw.Draw(img)
                draw.rectangle(((staff.ulx, staff.uly), (staff.lrx, staff.lry)), outline='red', fill=None, width=5)
                del draw
    scale = max_height / img.size[1]
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    return img


def draw_page(detector):
    img = construct_page_overlay(detector)
    img.show()


def save_annotations(detectors, score_name, suffix=''):
    annotations = ''
    for detector in detectors:
        page = detector.page
        system_annotations = list(map(lambda system: str(len(system.staff_boundaries)) + ',' + str(len(system.measures)), page.systems))
        annotations += ' '.join(system_annotations) + '\n'
    file_path = Path(data_dir, score_name, 'annotation_results' + suffix + '.txt').resolve()
    with open(file_path, 'w') as f:
        f.write(annotations)


def save_pages(detectors, score_name, suffix=''):
    pdf_filename = Path(data_dir, score_name, 'segmented_visual' + suffix + '.pdf').resolve()
    im1 = construct_page_overlay(detectors[0])
    im_list = []
    for i in range(1, len(detectors)):
        img = construct_page_overlay(detectors[i])
        im_list.append(img)
    im1.save(pdf_filename, 'PDF', resolution=100.0, save_all=True, append_images=im_list)


def detect(score_name, sys_method='lines', measure_method='region'):
    # page_path = Path(data_dir, score_name, 'ppm-300').resolve()
    page_path = Path(tmp_dir, 'single').resolve()
    paths = get_sorted_page_paths(page_path)
    detectors = []
    for i, path in tqdm(enumerate(paths)):
        print(path)
        detector = MeasureDetector(path).detect(plot=True, sys_method=sys_method, measure_method=measure_method)
        draw_page(detector)
        detectors.append(detector)
    if score_name != '':
        save_annotations(detectors, score_name, suffix='_'+sys_method)
        save_pages(detectors, score_name, suffix='_'+sys_method)


if __name__ == '__main__':
    # scores = ['Beethoven_Sextet', 'Beethoven_Septett', 'Debussy_La_Mer', 'Dukas_l\'Apprenti_Sorcier', 'Haydn_Symphony_104_London', 'Mendelssohn_Psalm_42', 'Mozart_Symphony_31', 'Schubert_Symphony_4', 'Van_Bree_Allegro']
    scores = ['']
    for score in scores:
        detect(score, sys_method='lines')

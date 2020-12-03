from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from posixpath import join
from collections import namedtuple
from segmenter.old.image_util import data_dir, tmp_dir
from segmenter.measure_detector import detect_measures

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


# Gives a sorted list of pages. As long as the pages are numbered incrementally separated with a "-", this will work fine.
def get_sorted_page_paths(page_path):
    paths = [p for p in Path(page_path).iterdir() if p.is_file()]
    return [str(p.resolve()) for p in sorted(paths, key=lambda p: int(str(p.stem).split("-")[1]))]


def compare_results(detected_pages, true_pages):
    outputs = []
    detected_system_count = 0
    detected_measure_count = 0
    detected_staff_count = 0
    total_systems = sum([len(p.systems) for p in true_pages])
    total_measures = sum([sum([s.measures for s in p.systems]) for p in true_pages])
    total_staffs = sum([sum([s.staffs for s in p.systems]) for p in true_pages])
    for i in range(len(detected_pages)):
        detected_system_count += len(detected_pages[i].systems)
        if len(detected_pages[i].systems) != len(true_pages[i].systems):
            outputs.append('Page {}: Expected {} systems, but got {}'.format(i, len(true_pages[i].systems), len(detected_pages[i].systems)))
    if detected_system_count == total_systems:
        for i in range(len(detected_pages)):
            detected_systems, true_systems = detected_pages[i].systems, true_pages[i].systems
            for j in range(len(detected_pages[i].systems)):
                detected_measure_count += len(detected_systems[j].measures)
                if len(detected_systems[j].measures) != true_systems[j].measures:
                    outputs.append('Page {}, system {}: Expected {} measures, but got {}'.format(i, j, true_systems[j].measures, len(detected_systems[j].measures)))
                system_staffs = list(map(lambda k: len(detected_systems[j].measures[k].staffs), range(len(detected_systems[j].measures))))
                if len(set(system_staffs)) > 1:
                    outputs.append('Page {}, system {}: Not all measures have equal numbers of staffs'.format(i, j))
                else:
                    detected_staff_count += system_staffs[0]
                    if system_staffs[0] != true_systems[j].staffs:
                        outputs.append('Page {}, system {}: Expected {} staffs, but got {}'.format(i, j, true_systems[j].staffs, len(detected_systems[j].measures[0].staffs)))
    print('Test results:\n=============')
    print('System score: {}/{} ({}%)'.format(detected_system_count, total_systems, round(detected_system_count/total_systems * 100)))
    print('Measure score: {}/{} ({}%)'.format(detected_measure_count, total_measures, round(detected_measure_count/total_measures * 100)))
    print('Staff score: {}/{} ({}%)'.format(detected_staff_count, total_staffs, round(detected_staff_count/total_staffs * 100)))
    print('\n'.join(outputs))


def build_true_system(system_str):
    properties = list(map(int, system_str.split(',')))
    return TrueSystem(staffs=properties[0], measures=properties[1])


def evaluate(score_name):
    with open(join(data_dir, score_name, 'annotations.txt')) as file:
        baseline = [list(map(build_true_system, line.rstrip().split(' '))) for line in file]
    true_pages = list(map(lambda s: TruePage(systems=s), baseline))

    page_paths = get_sorted_page_paths(join(data_dir, score_name, 'ppm-300'))
    if len(page_paths) != len(true_pages):
        print('Expected {} pages, but got {}'.format(len(true_pages), len(page_paths)))
        return

    pages = []
    for i, path in tqdm(enumerate(page_paths)):
        page = detect_measures(path)
        pages.append(page)
    compare_results(pages, true_pages)


def construct_page_overlay(img, page):
    max_height = 950
    for system in page.systems:
        for measure in system.measures:
            draw = ImageDraw.Draw(img)
            draw.rectangle(((measure.ulx, measure.uly), (measure.lrx, measure.lry)), outline='blue', fill=None,
                           width=10)
            del draw
            for staff in measure.staffs:
                draw = ImageDraw.Draw(img)
                draw.rectangle(((staff.ulx, staff.uly), (staff.lrx, staff.lry)), outline='red', fill=None, width=5)
                del draw
    scale = max_height / img.size[1]
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    return img


def draw_page(img, page):
    img = construct_page_overlay(img, page)
    img.show()


def save_pages(images, pages, score_name):
    pdf_filename = Path(data_dir, score_name, 'segmented_visual.pdf').resolve()
    im1 = construct_page_overlay(images[0], pages[0])
    im_list = []
    for i in range(1, len(images)):
        img = construct_page_overlay(images[i], pages[i])
        im_list.append(img)
    im1.save(pdf_filename, 'PDF', resolution=100.0, save_all=True, append_images=im_list)


def detect(score_name):
    page_path = Path(data_dir, score_name, 'ppm-300').resolve()
    # page_path = Path(tmp_dir, 'single').resolve()
    paths = get_sorted_page_paths(page_path)
    originals = []
    pages = []
    for i, path in tqdm(enumerate(paths)):
        original = Image.open(path)
        page = detect_measures(path, plot=False)
        # draw_page(original, page)
        originals.append(original.rotate(page.rotation, fillcolor='white'))
        pages.append(page)
    save_pages(originals, pages, score_name)


if __name__ == '__main__':
    scores = ['Van_Bree_Allegro']
    for score in scores:
        # evaluate(score)
        detect(score)

from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from segmenter.old.image_util import data_dir, tmp_dir
from segmenter.measure_detector import detect_measures
from util.files import get_sorted_page_paths

TrueSystem = namedtuple('TrueSystem', ['staffs', 'measures'])
TruePage = namedtuple('TruePage', ['systems'])


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
    scores = ['Beethoven_Sextet']
    for score in scores:
        detect(score)

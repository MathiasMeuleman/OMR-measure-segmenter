from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from posixpath import join
from segmenter.old.image_util import data_dir
from segmenter.measure_detector import detect_measures

score = 'Beethoven_Sextet'


# Gives a sorted list of pages. As long as the pages are numbered incrementally separated with a "-", this will work fine.
def get_sorted_page_paths(page_path):
    paths = [p for p in Path(page_path).iterdir() if p.is_file()]
    return [str(p.resolve()) for p in sorted(paths, key=lambda p: int(str(p.stem).split("-")[1]))]


def compare_results(expected, true, debug=False):
    if len(true) != len(expected):
        print('Expected {} pages, but got {}'.format(len(expected), len(true)))
        return
    outputs = []
    correct_system_count = 0
    correct_measure_count = 0
    for i in range(len(true)):
        if len(true[i]) == len(expected[i]):
            correct_system_count += 1
        else:
            outputs.append('Page {}: Expected {} systems, but got {}'.format(i, len(expected[i]), len(true[i])))
    if correct_system_count == len(true):
        for i in range(len(true)):
            for j in range(len(true[i])):
                if true[i][j] == expected[i][j]:
                    correct_measure_count += 1
                else:
                    outputs.append('Page {}, system {}: Expected {} measures, but got {}'.format(i, j, expected[i][j], true[i][j]))
    print('Test results:\n=============')
    print('System score: {}/{} ({}%)'.format(correct_system_count, len(true), round(correct_system_count/len(true) * 100)))
    print('Measure score: {}/{} ({}%)'.format(correct_measure_count, sum([len(l) for l in true]), round(correct_measure_count/sum([len(l) for l in true]) * 100)))
    print('\n'.join(outputs))


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


def save_pages(images, pages):
    pdf_filename = Path(data_dir, score, 'segmented_visual.pdf').resolve()
    im1 = construct_page_overlay(images[0], pages[0])
    im_list = []
    for i in range(1, len(images)):
        img = construct_page_overlay(images[i], pages[i])
        im_list.append(img)
    im1.save(pdf_filename, 'PDF', resolution=100.0, save_all=True, append_images=im_list)


def main():
    with open(join(data_dir, score, 'annotations_5.txt')) as file:
        baseline = [list(map(int, line.rstrip().split(' '))) for line in file]

    page_path = r"../tmp/test"
    paths = get_sorted_page_paths(page_path)
    originals = []
    pages = []
    for i, path in tqdm(enumerate(paths)):
        original = Image.open(path)
        page = detect_measures(path)
        # draw_page(original, page)
        originals.append(original)
        pages.append(page)
    save_pages(originals, pages)


if __name__ == '__main__':
    main()

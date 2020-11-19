from PIL import Image, ImageOps
from util.cv2_util import get_hough_angle
from segmenter.measure_detector import get_sorted_page_paths, open_and_preprocess
import numpy as np


def test():
    max_height = 950
    page_path = r"../../data/Beethoven_Sextet/ppm-600"
    paths = get_sorted_page_paths(page_path)
    for path in paths:
        img = open_and_preprocess(path)
        # original = Image.open(path)
        # systems = find_systems_in_score(img, plot=False)
        # for system in systems:
            # draw = ImageDraw.Draw(original)
            # blocks = find_blocks_in_system(img, system, plot=True)
        #     for block in blocks:
        #         draw = ImageDraw.Draw(original)
        #         draw.rectangle(((block.start, system.top), (block.end, system.bottom)), outline='red', fill=None, width=5)
        #         del draw
        # scale = max_height / original.size[1]
        # original = original.resize((int(original.size[0] * scale), int(original.size[1] * scale)), Image.ANTIALIAS)
        # original.show()


def show_img(img):
    img = Image.fromarray(img).convert('L')
    print(img.size)
    scale = 950 / img.size[1]
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    img.show()


def show_binary_img(img):
    show_img(img * 255)


def main():
    path = '../../data/Beethoven_Sextet/ppm-600/transcript-6.png'
    original = Image.open(path).convert('L')
    img = ImageOps.autocontrast(original)
    img = ImageOps.invert(img)
    img = np.asarray(img.point(lambda x: x > 50))
    angle2, display_img = get_hough_angle(np.array(img))
    print(angle2)
    print(display_img.shape)
    # try:
    #     angle1 = get_hough_angle(img)
    # except:
    #     angle1 = False
    # draw = ImageDraw.Draw(original)
    # draw.rectangle((area[0], area[1]))
    show_img(display_img)


if __name__ == "__main__":
    main()

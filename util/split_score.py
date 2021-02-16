from PIL import Image
from pathlib import Path
from segmenter.dirs import data_dir
from util.PIL_util import resize_img


def split_img_vertical(img_path):
    img = Image.open(img_path)
    w,h = img.size
    left = img.crop((0, 0, w//2, h))
    right = img.crop((w//2, 0, w, h))
    return [left, right]


def get_sorted_page_paths(page_path):
    paths = [p for p in Path(page_path).iterdir() if p.is_file()]
    return [str(p.resolve()) for p in sorted(paths, key=lambda p: int(str(p.stem).split("-")[1]))]


def split_pages(folder, out_path):
    Path(out_path).mkdir(exist_ok=True)
    paths = get_sorted_page_paths(folder)
    for i, path in enumerate(paths):
        imgs = split_img_vertical(path)
        imgs[0].save(Path(out_path, 'transcript-{}.png'.format((i+1)*2 - 1)))
        imgs[1].save(Path(out_path, 'transcript-{}.png'.format((i+1)*2)))


if __name__ == '__main__':
    score = 'Van_Bree_Allegro'
    folder = Path(data_dir, score, 'ppm-300-pre')
    out_path = Path(data_dir, score, 'ppm-300')
    split_pages(folder, out_path)

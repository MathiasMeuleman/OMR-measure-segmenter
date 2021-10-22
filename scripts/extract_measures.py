from PIL import Image
from tqdm import tqdm

from score_analysis.measure_extractor import MeasureExtractor, filter_pages
from util.dirs import get_musicdata_scores, page_sort_key


def list_filtered_page_differences():
    musicdata_paths = get_musicdata_scores()
    total_pages = 0
    total_filtered_pages = 0
    for part in musicdata_paths:
        pages = sorted((part / 'pages').iterdir(), key=page_sort_key)
        filtered_pages = filter_pages(part)
        total_pages += len(pages)
        total_filtered_pages += len(filtered_pages)
        print('Part: {}/{}, {} / {} pages selected'.format(part.parent.name, part.name, len(filtered_pages), len(pages)))
    print('{} / {} pages selected ({}%)'.format(total_filtered_pages, total_pages, round(total_filtered_pages / total_pages * 100, 2)))


if __name__ == '__main__':
    list_filtered_page_differences()
    musicdata_paths = get_musicdata_scores()
    for part in musicdata_paths:
        measures_path = part / 'measures'
        for page in tqdm(filter_pages(part)):
            measure_images_path = part / 'measure_images' / page.stem
            measure_images_path.mkdir(parents=True, exist_ok=True)
            measure_path = measures_path / (page.stem + '.json')
            MeasureExtractor(Image.open(page), measure_path, measure_images_path).extract_measures()

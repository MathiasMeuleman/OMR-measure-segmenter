from PIL import Image
from tqdm import tqdm

from score_analysis.measure_detector import MeasureDetector
from util.dirs import musicdata_dir
from util.pdf2png import convert_pdf_file
from util.score_draw import ScoreDraw


def get_score_path(directory):
    score_path = next((f for f in directory.iterdir() if f.suffix == '.pdf'), None)
    if not score_path.is_file():
        raise FileNotFoundError('Could not find PDF score file in directory ' + str(directory))
    return score_path


def verify_pages_exist(directory):
    pages_path = directory / 'pages'
    if not pages_path.is_dir():
        pages_path.mkdir()
    if not (pages_path / 'page_1.png').is_file():
        score_path = get_score_path(directory)
        convert_pdf_file(score_path, pages_path)


def detect_score_measures(directory):
    verify_pages_exist(directory)
    staff_path = directory / 'staffs'
    staff_path.mkdir(exist_ok=True, parents=True)
    barline_path = directory / 'barlines'
    barline_path.mkdir(exist_ok=True, parents=True)
    measures_path = directory / 'measures'
    measures_path.mkdir(exist_ok=True, parents=True)
    overlay_path = directory / 'measure_overlays'
    overlay_path.mkdir(exist_ok=True, parents=True)

    print('Detecting measures in {}/{}'.format(directory.parents[0].name, directory.name))

    for page in tqdm([page for page in (directory / 'pages').iterdir() if page.suffix == '.png']):
        json_name = page.stem + '.json'
        measure_detector = MeasureDetector(page, staff_path=staff_path / json_name,
                                           barline_path=barline_path / json_name, output_path=measures_path / json_name)
        measures = measure_detector.detect_measures(force_barlines=True)
        score_draw = ScoreDraw(Image.open(page))
        image = score_draw.draw_measures(measures)
        image.save(overlay_path / page.name)


if __name__ == '__main__':
    directories = [
        'bach_brandenburg_concerto_5_part_1',
        'beethoven_symphony_1/part_1',
        'beethoven_symphony_1/part_2',
        'beethoven_symphony_1/part_3',
        'beethoven_symphony_1/part_4',
        'beethoven_symphony_2/part_1',
        'beethoven_symphony_2/part_2',
        'beethoven_symphony_2/part_3',
        'beethoven_symphony_2/part_4',
        'beethoven_symphony_3/part_1',
        'beethoven_symphony_3/part_2',
        'beethoven_symphony_3/part_3',
        'beethoven_symphony_3/part_4',
        'beethoven_symphony_4/part_1',
        'beethoven_symphony_4/part_2',
        'beethoven_symphony_5/part_1',
        'beethoven_symphony_5/part_2',
        'beethoven_symphony_5/part_3',
        'beethoven_symphony_5/part_4',
        'beethoven_symphony_6/part_1',
        'beethoven_symphony_6/part_2',
        'beethoven_symphony_6/part_3',
        'beethoven_symphony_6/part_4',
        'beethoven_symphony_6/part_5',
        'beethoven_symphony_7/part_1',
        'beethoven_symphony_7/part_2',
        'beethoven_symphony_7/part_3',
        'beethoven_symphony_7/part_4',
        'beethoven_symphony_8/part_1',
        'beethoven_symphony_8/part_2',
        'beethoven_symphony_8/part_3',
        'beethoven_symphony_8/part_4',
        'beethoven_symphony_9/part_1',
        'beethoven_symphony_9/part_2',
        'beethoven_symphony_9/part_3',
        'beethoven_symphony_9/part_4',
        'brahms_symphony_3',
        'bruckner_symphony_5',
        'bruckner_symphony_9',
        'holst_the_planets',
        'mahler_symphony_4',
        'mozart_symphony_41',
        'tchaikovsky_ouverture_1812',
    ]
    for directory in directories:
        detect_score_measures(musicdata_dir / directory)

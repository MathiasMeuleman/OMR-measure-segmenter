from PIL import Image
from tqdm import tqdm

from score_analysis.measure_detector import MeasureDetector
from util.dirs import musicdata_dir, get_musicdata_scores
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
    staff_overlay_path = directory / 'staff_overlays'
    staff_overlay_path.mkdir(exist_ok=True, parents=True)
    barline_path = directory / 'barlines'
    barline_path.mkdir(exist_ok=True, parents=True)
    barline_overlay_path = directory / 'barline_overlays'
    barline_overlay_path.mkdir(exist_ok=True, parents=True)
    measures_path = directory / 'measures'
    measures_path.mkdir(exist_ok=True, parents=True)
    overlay_path = directory / 'measure_overlays'
    overlay_path.mkdir(exist_ok=True, parents=True)

    print('Detecting measures in {}/{}'.format(directory.parents[0].name, directory.name))
    pages = [page for page in (directory / 'pages').iterdir() if page.suffix == '.png']
    for page in tqdm(pages):
        json_name = page.stem + '.json'
        page_staff_path = staff_path / json_name
        page_barline_path = barline_path / json_name
        measure_detector = MeasureDetector(page, staff_path=page_staff_path,
                                           barline_path=page_barline_path, output_path=measures_path / json_name)
        measure_detector.detect_measures(force_barlines=True)
        score_draw = ScoreDraw(Image.open(page))

        measure_image = score_draw.draw_measures(measure_detector.measures)
        measure_image.save(overlay_path / page.name)
        staff_image = score_draw.draw_staffs(measure_detector.staffs)
        staff_image.save(staff_overlay_path / page.name)
        system_image = score_draw.draw_systems(measure_detector.systems)
        system_image.save(barline_overlay_path / page.name)


if __name__ == '__main__':
    for directory in get_musicdata_scores():
        detect_score_measures(musicdata_dir / directory)

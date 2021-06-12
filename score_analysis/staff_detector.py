import subprocess
from pathlib import Path
from tqdm import tqdm
from util.pdf2png import convert_pdf_file


class StaffDetector:

    def __init__(self, directory):
        self.directory = directory

    def get_score_path(self):
        score_path = next((f for f in Path(self.directory).iterdir() if f.suffix == '.pdf'), None)
        if not score_path.is_file():
            raise FileNotFoundError('Could not find PDF score file in directory ' + str(self.directory))
        return score_path

    def verify_pages_exist(self):
        score_path = self.get_score_path()
        pages_path = Path(self.directory) / 'pages'
        if not pages_path.is_dir():
            pages_path.mkdir()
        if not (pages_path / 'page_1.png').is_file():
            convert_pdf_file(score_path, pages_path)

    def run_py2_detect_staffs(self, image_path, output_path):
        detect_staffs_path = (Path(__file__).parent / 'py2_detect_staffs.py').absolute()
        subprocess.run(['/usr/bin/python', detect_staffs_path, image_path.absolute(), output_path.absolute()])

    def detect_staffs(self):
        self.verify_pages_exist()
        staffs_path = Path(self.directory) / 'staffs'
        pages_path = Path(self.directory) / 'pages'
        if not staffs_path.is_dir():
            staffs_path.mkdir()
        for i in tqdm(range(len(list(pages_path.iterdir())))):
            self.run_py2_detect_staffs(pages_path / 'page_{}.png'.format(i+1), staffs_path / 'page_{}.json'.format(i+1))


if __name__ == '__main__':
    data_dir = Path(__file__).absolute().parent.parent.parent / 'OMR-measure-segmenter-data/musicdata'
    score_dir = data_dir / 'bach_brandenburg_concerto_5_part_1'
    StaffDetector(score_dir).detect_staffs()

from score_analysis.staff_detector import StaffDetector
from score_analysis.staff_verifyer import StaffVerifyer
from pathlib import Path

if __name__ == '__main__':
    data_dir = Path(__file__).absolute().parent.parent.parent / 'OMR-measure-segmenter-data/musicdata'
    score_dir = data_dir / 'tchaikovsky_ouverture_1812/edition_1'
    # score_dir = data_dir / 'mozart_symphony_41'
    # page = 5
    # StaffDetector(score_dir).run_py2_detect_staffs(score_dir / 'pages' / 'page_{}.png'.format(page), score_dir / 'staffs' / 'page_{}.json'.format(page))
    # StaffDetector(score_dir).detect_staffs()
    verifyer = StaffVerifyer(score_dir)
    # verifyer.overlay_page_staffs(page)
    # verifyer.overlay_staffs()
    verifyer.verify_staffs()

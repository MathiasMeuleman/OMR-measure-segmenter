from score_analysis.staff_detector import StaffDetector

from util.dirs import data_dir

if __name__ == '__main__':
    score_dir = data_dir / 'sample'
    StaffDetector(score_dir, 'Meuleman').detect_staffs()

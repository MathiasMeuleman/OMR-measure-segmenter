from score_analysis.staff_detector import StaffDetector
from score_analysis.staff_verifyer import StaffVerifyer

from util.dirs import data_dir

if __name__ == '__main__':
    score_dir = data_dir / 'sample'
    staff_finder = 'Meuleman'
    staffs = StaffDetector(score_dir, staff_finder).detect_staffs()
    staffs_path = score_dir / 'staffs' / staff_finder
    overlay_path = score_dir / 'staff_overlays' / staff_finder
    verifyer = StaffVerifyer(score_dir, staffs, staffs_path=staffs_path, overlay_path=overlay_path)
    verifyer.overlay_staffs()

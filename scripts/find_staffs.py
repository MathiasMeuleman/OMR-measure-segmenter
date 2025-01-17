from score_analysis.staff_detector import StaffDetector
from score_analysis.staff_verifyer import StaffVerifyer

from util.dirs import musicdata_dir

if __name__ == '__main__':
    score_dirs = [dir for dir in musicdata_dir.iterdir() if dir.is_dir()]
    part_dirs = [list(dir.glob('part_*/')) if dir.name.startswith('beethoven') else [dir] for dir in score_dirs]
    part_dirs = [part for parts in part_dirs for part in parts]
    for score_dir in part_dirs:
        for staff_finder in ['Meuleman']:
            staffs = StaffDetector(score_dir, staff_finder).detect_staffs()
            staffs_path = score_dir / 'staffs' / staff_finder
            overlay_path = score_dir / 'staff_overlays' / staff_finder
            verifyer = StaffVerifyer(score_dir, staffs, staffs_path=staffs_path, overlay_path=overlay_path)
            verifyer.generate_staff_overlays_pdf()

import json
from PIL import Image

from score_analysis.barline_detector import System
from score_analysis.staff_detector import Staff
from util.dirs import musicdata_dir
from util.score_draw import ScoreDraw

part = musicdata_dir / 'beethoven_symphony_1' / 'part_1'
page = 3
with open(part / 'barlines' / 'page_{}.json'.format(page)) as f:
    systems = [System.from_json(system) for system in json.load(f)['systems']]
with open(part / 'staffs' / 'page_{}.json'.format(page)) as f:
    staffs = [Staff.from_json(staff) for staff in json.load(f)['staffs']]
image = Image.open(part / 'pages' / 'page_{}.png'.format(page))
# image = ScoreDraw(image).draw_systems(systems)
image = ScoreDraw(image).draw_staffs(staffs)
image.save('staffs.png')

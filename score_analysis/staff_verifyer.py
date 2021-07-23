import json
from PIL import Image, ImageColor, ImageDraw
from pathlib import Path
from tqdm import tqdm


class StaffVerifyer:

    def __init__(self, directory):
        self.directory = directory
        self.mapping_path = Path(directory) / 'score_mapping.json'
        self.staffs_path = Path(directory) / 'staffs'
        self.pages_path = Path(directory) / 'pages'
        self.overlay_path = Path(directory) / 'staff_overlays'

    def overlay_page_staffs(self, pagenumber):
        page_name = 'page_{}'.format(pagenumber)
        with open(self.staffs_path / '{}.json'.format(page_name)) as file:
            staves = json.load(file)
        img = Image.open(self.pages_path / '{}.png'.format(page_name))

        max_height = 950
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for j, staff in enumerate(staves['staves']):
            color = ImageColor.getrgb(colors[j % len(colors)])
            for line in staff['lines']:
                draw = ImageDraw.Draw(img, mode='RGBA')
                draw.line([tuple(point) for point in line], fill=color, width=5)
                del draw
        scale = max_height / img.size[1]
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
        img.save(self.overlay_path / 'pages' / '{}.png'.format(page_name))
        return img

    def overlay_staffs(self):
        overlay_pages_path = self.overlay_path / 'pages'
        if not overlay_pages_path.is_dir():
            overlay_pages_path.mkdir(parents=True)
        if not self.staffs_path.is_dir():
            raise AssertionError('Could not find staffs directory')
        images = []
        for i in tqdm(range(len(list(self.staffs_path.iterdir())))):
            image = self.overlay_page_staffs(i + 1)
            images.append(image)
        images[0].save(self.overlay_path / 'staff_overlays.pdf', 'PDF', resolution=100.0, save_all=True, append_images=images[1:])

    def verify_staffs(self):
        with open(self.mapping_path) as file:
            mapping = json.load(file)
        for i, page in enumerate(mapping['pages']):
            with open(self.staffs_path / 'page_{}.json'.format(i + 1)) as file:
                page_staffs = json.load(file)
            true_staff_count = sum([1 for system in page['systems'] for staff in system['staffs']])
            found_staff_count = sum([1 for staff in page_staffs['staves']])
            deviation = '\t!!' if true_staff_count != found_staff_count else ''
            print('Page {}\tFound: {}\tExpected: {}{}'.format(i + 1, found_staff_count, true_staff_count, deviation))
            found_staffline_count = [len(stave['lines']) for stave in page_staffs['staves']]
            unexpected_stafflines = [i for i in range(len(found_staffline_count)) if found_staffline_count[i] not in [1, 5]]
            if len(unexpected_stafflines) > 0:
                print('\tFound staffline count: {}'.format(['{}: {}'.format(i, found_staffline_count[i]) for i in unexpected_stafflines]))


if __name__ == '__main__':
    data_dir = Path(__file__).absolute().parent.parent.parent / 'OMR-measure-segmenter-data/musicdata'
    score_dir = data_dir / 'bach_brandenburg_concerto_5_part_1'
    StaffVerifyer(score_dir).overlay_page_staffs(16)

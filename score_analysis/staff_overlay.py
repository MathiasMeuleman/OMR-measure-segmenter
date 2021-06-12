import json
from PIL import Image, ImageColor, ImageDraw
from pathlib import Path
from tqdm import tqdm


class StaffOverlay:

    def __init__(self, directory):
        self.directory = directory
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
        images[0].save(self.overlay_path / 'staff_overlays.pdf', 'PDF', resolution=100.0, save_all=True, append_images=images[1:-1])


if __name__ == '__main__':
    data_dir = Path(__file__).absolute().parent.parent.parent / 'OMR-measure-segmenter-data/musicdata'
    score_dir = data_dir / 'bach_brandenburg_concerto_5_part_1'
    StaffOverlay(score_dir).overlay_staffs()

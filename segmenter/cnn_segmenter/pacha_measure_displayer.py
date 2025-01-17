import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

from util.dirs import data_dir


def display_measures(pages):
    path = data_dir / 'measure_annotations-Mahler1-part3.xml'
    root = ET.parse(path).getroot()
    facsimile = root[1][0]

    for idx in pages:
        page = facsimile[idx]
        page_iter = iter(page)
        image_path = data_dir / 'Mahler_Symphony_1/ppm-300' / next(page_iter).attrib['target']
        img = Image.open(image_path)
        scalex = int(page.attrib['lrx']) / img.size[0]
        scaley = int(page.attrib['lry']) / img.size[1]
        for zone in page_iter:
            ulx = int(zone.attrib['ulx'])
            uly = int(zone.attrib['uly'])
            lrx = int(zone.attrib['lrx'])
            lry = int(zone.attrib['lry'])
            draw = ImageDraw.Draw(img)
            draw.rectangle([(ulx/scalex, uly/scaley), (lrx/scalex, lry/scaley)], outline='green', fill=None, width=5)
            del draw
        resize = 18
        dims = (int(img.size[1] * resize/100), int(img.size[0] * resize/100))
        img.thumbnail(dims, Image.ANTIALIAS)
        img.show()
        img.save('pacha_{}.png'.format(76+idx), "PNG")


if __name__ == "__main__":
    display_measures([])

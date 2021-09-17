import tempfile
import pdf2image
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PyPDF2 import PdfFileReader
from pathlib import Path
from tqdm import tqdm

from util.dirs import musicdata_dir


def convert_pdf_file(in_path, out_path, first_page=1, last_page=-1):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    pdf = PdfFileReader(open(in_path, 'rb'))
    max_pages = pdf.numPages
    if last_page > 0:
        max_pages = min(max_pages, last_page)
    index = first_page
    for page in tqdm(range(first_page, max_pages + 1, 10)):
        with tempfile.TemporaryDirectory() as outpath:
            images = pdf2image.convert_from_path(in_path, dpi=300, output_folder=outpath, first_page=page, last_page=min(page + 10 - 1, max_pages))
            for image in images:
                image.save(out_path / 'page_{}.png'.format(index))
                index += 1


if __name__ == '__main__':
    source_path = musicdata_dir / 'beethoven_symphony_2/IMSLP503997-PMLP2580-combinepdf.pdf'
    convert_pdf_file(source_path, out_path=musicdata_dir / 'beethoven_symphony_2/pages')

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


def convert_pdf_file(_path, out_path, first_page=1, last_page=-1):
    Path(out_path).mkdir(exist_ok=True)
    pdf = PdfFileReader(open(_path, 'rb'))
    max_pages = pdf.numPages
    if last_page > 0:
        max_pages = min(max_pages, last_page)
    index = first_page
    for page in tqdm(range(first_page, max_pages, 10)):
        with tempfile.TemporaryDirectory() as outpath:
            images = pdf2image.convert_from_path(_path, dpi=300, output_folder=outpath, first_page=page, last_page=min(page + 10 - 1, max_pages))
            for image in images:
                image.save(str((Path(out_path) / "transcript-{0}.png".format(index)).resolve()))
                index += 1


if __name__ == "__main__":
    path = r"../data/Van_Bree_Allegro/IMSLP82485-PMLP136878-1602.Bree-Allegro4SQs.sc.pdf"
    convert_pdf_file(path, out_path=r"../data/Van_Bree_Allegro/ppm-300")

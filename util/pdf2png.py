import tempfile
import pdf2image
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PyPDF2 import PdfFileReader
from pathlib import Path


def convert_pdf_file(_path, out_path, first_page=1, last_page=-1):
    pdf = PdfFileReader(open(_path, 'rb'))
    max_pages = pdf.numPages
    if last_page > 0:
        max_pages = min(max_pages, last_page)
    index = first_page
    for page in range(first_page, max_pages, 10):
        with tempfile.TemporaryDirectory() as outpath:
            images = pdf2image.convert_from_path(_path, dpi=600, output_folder=outpath, first_page=page, last_page=min(page + 10 - 1, max_pages))
            for image in images:
                image.save(str((Path(out_path) / "transcript-{0}.png".format(index)).resolve()))
                index += 1


if __name__ == "__main__":
    path = r"../data/Beethoven_Andante_FMajor/IMSLP109835-PMLP30951-sibley.1802.1525.beethoven.andante.fmajor.pdf"
    convert_pdf_file(path, out_path=r"../data/Beethoven_Andante_FMajor/ppm-600")

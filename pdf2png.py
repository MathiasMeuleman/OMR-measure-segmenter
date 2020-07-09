import tempfile
import pdf2image
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PyPDF2 import PdfFileReader


def convert_pdf_file(_path, first_page=1, last_page=-1):
    pdf = PdfFileReader(open(_path, 'rb'))
    max_pages = pdf.numPages
    if last_page > 0:
        max_pages = min(max_pages, last_page)
    index = first_page
    for page in range(first_page, max_pages, 10):
        with tempfile.TemporaryDirectory() as outpath:
            images = pdf2image.convert_from_path(_path, dpi=600, output_folder=outpath, first_page=page, last_page=min(page + 10 - 1, max_pages))
            for image in images:
                image.save("transcript-{0}.png".format(index))
                index += 1


if __name__ == "__main__":
    path = r'C:\Users\Mathias\Documents\TU Delft\Thesis\thesis\data\IMSLP17070-Mahler-Symph1fs.pdf'
    convert_pdf_file(path)

import pathlib
import os
import certifi
from pdf2image import convert_from_path

# Fix SSL issues on Windows
os.environ["SSL_CERT_FILE"] = certifi.where()


class Extractor:
    def __init__(self):
        # Input / Output folders
        self.PDF_FOLDER = pathlib.Path("PDF")
        self.IMG_FOLDER = pathlib.Path("images")

        # PDF → Image settings
        self.DPI = 500
        self.IMG_FORMAT = "PNG"

        # Poppler path (Windows)
        self.poppler_path = r"D:\poppler-25.07.0\Library\bin"

        # Ensure output folder exists
        self.IMG_FOLDER.mkdir(exist_ok=True)

    # ===============================================================
    # PDF → list of PIL images
    # ===============================================================
    def pdf_to_images(self, pdf_path):
        return convert_from_path(
            str(pdf_path),
            poppler_path=self.poppler_path,
            dpi=self.DPI
        )

    # ===============================================================
    # Save images for ONE pdf
    # ===============================================================
    def save_all_images(self, pdf_path):
        pages = self.pdf_to_images(pdf_path)

        pdf_name = pdf_path.stem
        out_dir = self.IMG_FOLDER / pdf_name
        out_dir.mkdir(exist_ok=True)

        for i, img in enumerate(pages, start=1):
            out_path = out_dir / f"page_{i}.png"
            img.save(out_path, self.IMG_FORMAT)

    # ===============================================================
    # Run all PDFs
    # ===============================================================
    def run(self):
        pdfs = list(self.PDF_FOLDER.glob("**/*.pdf"))

        if not pdfs:
            return

        for pdf in pdfs:
            self.save_all_images(pdf)


def main():
    extractor = Extractor()
    extractor.run()


if __name__ == "__main__":
    main()

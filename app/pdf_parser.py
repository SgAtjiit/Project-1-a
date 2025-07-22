import fitz # PyMuPDF

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_text_blocks(self):
        """
        Extracts text blocks with their properties (text, font size, bbox, page number).
        Returns a list of dictionaries, each representing a text block.
        """
        text_blocks = []
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"] # "dict" format provides detailed info

            for b in blocks:
                if b["type"] == 0: # text block
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text_block = {
                                "text": span["text"].strip(),
                                "font_size": round(span["size"], 1),
                                "font_name": span["font"],
                                "bbox": span["bbox"], # (x0, y0, x1, y1)
                                "page": page_num + 1 # 1-indexed page number
                            }
                            if text_block["text"]: # Only add non-empty text
                                text_blocks.append(text_block)
        return text_blocks

    def get_title_candidate(self):
        """
        Attempts to find a title. A common heuristic is the largest font size
        on the first page, or the first prominent text.
        This is a heuristic and can be improved with ML if a training set is available.
        """
        if len(self.doc) > 0:
            first_page = self.doc.load_page(0)
            text_on_first_page = first_page.get_text("dict")["blocks"]

            if not text_on_first_page:
                return None

            # Simple heuristic: find the text with the largest font size on the first page
            # This can be refined significantly.
            largest_font_text = ""
            max_font_size = 0.0

            for b in text_on_first_page:
                if b["type"] == 0:
                    for line in b["lines"]:
                        for span in line["spans"]:
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                largest_font_text = span["text"].strip()
                                # Consider only text at the top of the page for title
                                if span["bbox"][1] < page.rect.height / 3: # Upper third of the page
                                    return largest_font_text
            # Fallback if no prominent text in upper third or for simple PDFs
            return largest_font_text if largest_font_text else None
        return None
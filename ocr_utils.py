# ocr_utils.py
import pdfplumber
from PIL import Image
import pytesseract
import os
import io

# If Tesseract not on PATH, set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf_with_pages(pdf_path, ocr_lang="eng"):
    """
    Returns a list of dicts: [{'page_num': 1, 'text': '...', 'ocr_confidence': 95.2, 'has_text_layer': True}, ...]
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            has_text_layer = bool(page_text and page_text.strip())
            if has_text_layer:
                # basic confidence: assume 100 for text layer (we can't get confidence from text layer)
                pages.append({
                    "page_num": i,
                    "text": page_text.strip(),
                    "ocr_confidence": 100.0,
                    "has_text_layer": True
                })
            else:
                # fallback OCR on raster image of page
                try:
                    pil_image = page.to_image(resolution=300).original
                except Exception:
                    # alternative: render page as PIL.Image via convert if needed
                    pil_image = page.to_image().original
                ocr_result = pytesseract.image_to_data(pil_image, lang=ocr_lang, output_type=pytesseract.Output.DICT)
                # Reconstruct text and compute average confidence
                words = []
                confs = []
                for w, c in zip(ocr_result.get('text', []), ocr_result.get('conf', [])):
                    if w.strip():
                        words.append(w)
                        try:
                            confs.append(float(c))
                        except:
                            pass
                page_text = " ".join(words).strip()
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                pages.append({
                    "page_num": i,
                    "text": page_text,
                    "ocr_confidence": avg_conf,
                    "has_text_layer": False
                })
    return pages


def extract_text_from_docx(docx_path):
    from docx import Document
    doc = Document(docx_path)
    texts = []
    for i, para in enumerate(doc.paragraphs, start=1):
        if para.text.strip():
            texts.append(para.text.strip())
    return "\n".join(texts)

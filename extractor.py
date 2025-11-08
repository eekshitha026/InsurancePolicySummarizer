import os
from pypdf import PdfReader

# The path where Streamlit saves uploaded files
UPLOAD_DIR = "uploads"
# The collection name for ChromaDB
CHROMA_COLLECTION_NAME = "insurance_policy_clauses"


def extract_text_from_pdf(pdf_path: str):
    """
    Extracts text page by page from a PDF file.
    Returns a list of strings, where each string is the text of a page.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return []

    try:
        reader = PdfReader(pdf_path)
        page_texts = []
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")

        return page_texts
    except Exception as e:
        print(f"An error occurred during PDF extraction: {e}")
        return []


def get_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in the PDF."""
    if not os.path.exists(pdf_path):
        return 0
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception:
        return 0

# ocr_utils.py is not strictly needed as modern PDFs have searchable text.
# We rely on pypdf and the powerful text processing of the Gemini API.
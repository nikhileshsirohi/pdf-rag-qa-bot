from pypdf import PdfReader

def load_pdf(pdf_path: str) -> str:
    """
    Load a PDF file and extract text from all pages.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """

    reader = PdfReader(pdf_path)
    full_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            full_text.append(text)
        else:
            print(f"Warning: No text found on page {page_num + 1}")
    print(f"Total pages in PDF: {len(reader.pages)}")
    return "\n".join(full_text)

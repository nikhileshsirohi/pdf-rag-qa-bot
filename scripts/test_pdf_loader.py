from app.pdf_loader import load_pdf

pdf_path = "data/raw_pdfs/sample.pdf"
text = load_pdf(pdf_path)

print(text[:1000])
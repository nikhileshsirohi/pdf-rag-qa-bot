from app.pdf_loader import load_pdf
from app.text_splitter import split_text

pdf_path = "data/raw_pdfs/sample.pdf"

text = load_pdf(pdf_path)
chunks = split_text(text, 500, 1)

print(f"Total Chunks Created: {len(chunks)}\n")

# for i, chunk in enumerate(chunks):
#     print(f"--- Chunk {i+1} ---")
#     print(chunk[:50])
#     print()
print(chunks[0])
print("...")
print(chunks[1])
print("...")
print(chunks[2])

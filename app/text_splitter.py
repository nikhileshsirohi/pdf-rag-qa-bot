import re #regular expressions module

def split_text(
    text: str,
    chunk_size: int = 500,
    overlap_paragraphs: int = 1
) -> list:
    """
    Robust paragraph-based chunking for PDFs with numbering.
    """

    # Normalize text
    text = re.sub(r'\n+', '\n', text)  # collapse multiple newlines
    text = text.strip()

    # Split on paragraph-like boundaries:
    # - blank lines
    # - numbered sections
    paragraphs = re.split(
        r'\n\s*\n|(?=\n\d+\.)|(?=\n[A-Z][a-z])',
        text
    )

    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > chunk_size:
            chunks.append("\n\n".join(current_chunk))

            # overlap last N paragraphs
            current_chunk = current_chunk[-overlap_paragraphs:]
            current_length = sum(len(p) for p in current_chunk)

        current_chunk.append(para)
        current_length += para_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    print(f"Total paragraphs: {len(paragraphs)}")
    return chunks

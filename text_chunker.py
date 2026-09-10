from pypdf import PdfReader


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
pdf_path = "documents/Advanced Social Text and Media Analytics.pdf"

text = extract_text(pdf_path)

chunks = chunk_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
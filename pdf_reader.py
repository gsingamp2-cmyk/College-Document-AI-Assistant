from pypdf import PdfReader

pdf_path = "documents/Advanced Social Text and Media Analytics.pdf"

reader = PdfReader(pdf_path)

print(f"Number of pages: {len(reader.pages)}")

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n--- Page {page_number} ---\n")
    print(text)
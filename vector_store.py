import chromadb
from sentence_transformers import SentenceTransformer
from text_chunker import extract_text, chunk_text

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent vector database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="college_documents"
)

# PDF path
pdf_path = "documents/Advanced Social Text and Media Analytics.pdf"

# Extract text
text = extract_text(pdf_path)

# Create chunks
chunks = chunk_text(text)

print("Total chunks:", len(chunks))

# Generate embeddings
embeddings = model.encode(chunks).tolist()

# Store chunks in ChromaDB
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=ids
)

print("Documents stored:", collection.count())
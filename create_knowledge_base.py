import os
import sys
from pathlib import Path
import PyPDF2
import chromadb
import re
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv('CHROMA_CLOUD_API_KEY')
TENANT = os.getenv('CHROMA_CLOUD_TENANT')
DATABASE = os.getenv('CHROMA_CLOUD_DATABASE', 'operating_system')
COLLECTION = os.getenv('CHROMA_COLLECTION_NAME', 'os-knowledge-base')
CHUNK_SIZE = 500


pdf_path = "Operating System Concepts.pdf"

print("=" * 60)
print("PDF to Chroma DB Uploader")
print("=" * 60)

# Extract text from PDF
print(f"\nExtracting text from: {pdf_path}")
try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        text = ""
        
        for i, page in enumerate(pdf_reader.pages, 1):
            text += page.extract_text() + "\n"
            if i % 10 == 0 or i == total_pages:
                print(f"   ✓ Processed {i}/{total_pages} pages")
    
    print(f"✓ Extracted {len(text)} characters")
except Exception as e:
    print(f"Error extracting PDF: {e}")
    sys.exit(1)

# Clean text
text = re.sub(r'\s+', ' ', text)
print(f"✓ Cleaned text")

# Create chunks
print(f"\nCreating chunks (size: {CHUNK_SIZE})...")
chunks = []
words = text.split()
current_chunk = ""

for word in words:
    if len(current_chunk) + len(word) < CHUNK_SIZE:
        current_chunk += word + " "
    else:
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        current_chunk = word + " "

if current_chunk.strip():
    chunks.append(current_chunk.strip())

print(f"✓ Created {len(chunks)} chunks")

# Connect to Chroma DB Cloud
print(f"\nConnecting to Chroma DB Cloud...")
try:
    client = chromadb.CloudClient(
        api_key=API_KEY,
        tenant=TENANT,
        database=DATABASE
    )
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ Connected to Chroma DB Cloud")
    print(f"   Collection: {COLLECTION}")
except Exception as e:
    print(f"Error connecting to Chroma DB: {e}")
    sys.exit(1)

# Upload chunks
print(f"\nUploading {len(chunks)} chunks...")
try:
    pdf_name = Path(pdf_path).stem
    
    for i, chunk in enumerate(chunks):
        doc_id = f"{pdf_name}_{i}"
        collection.add(
            ids=[doc_id],
            documents=[chunk],
            metadatas=[{
                "source": Path(pdf_path).name,
                "chunk": f"{i+1}/{len(chunks)}"
            }]
        )
        
        if (i + 1) % 50 == 0 or i == len(chunks) - 1:
            print(f"   ✓ Uploaded {i+1}/{len(chunks)} chunks")
    
    print(f"✓ All chunks uploaded successfully!")
    
except Exception as e:
    print(f"Error uploading chunks: {e}")
    sys.exit(1)

# Show statistics
print("\n" + "=" * 60)
print("✓ SUCCESS!")
print("=" * 60)
print(f"PDF: {pdf_path}")
print(f"Total chunks: {len(chunks)}")
print(f"Collection: {COLLECTION}")
print(f"Database: {DATABASE}")
print("=" * 60)

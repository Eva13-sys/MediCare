from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print(" Loading PDF files...")
extracted_data = load_pdf_file(data='data/')

print(f" Loaded {len(extracted_data)} documents")

print(" Filtering documents...")
filter_data = filter_to_minimal_docs(extracted_data)

print(" Splitting text into chunks...")
text_chunks = text_split(filter_data)

print(f" Created {len(text_chunks)} chunks")

print(" Loading embeddings...")
embeddings = download_embeddings()

print(" Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicare"

if not pc.has_index(index_name):
    print(f" Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f" Index '{index_name}' already exists")

index = pc.Index(index_name)

print(" Clearing old vectors from index...")
index.delete(delete_all=True)

print(f" Uploading {len(text_chunks)} chunks to Pinecone...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(" Upload complete!")

stats = index.describe_index_stats()
print(f" Total vectors in index: {stats['total_vector_count']}")
from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings  # ← Changed name
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print(" Loading PDF files...")
extracted_data = load_pdf_file(data='data/')

print(" Filtering documents...")
filter_data = filter_to_minimal_docs(extracted_data)

print(" Splitting text into chunks...")
text_chunks = text_split(filter_data)

print(" Loading embeddings...")
embeddings = download_embeddings()

print(" Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicare"

if not pc.has_index(index_name):
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f" Index '{index_name}' already exists")

index = pc.Index(index_name)

stats = index.describe_index_stats()
if stats['total_vector_count'] > 0:
    print(f" Index already has {stats['total_vector_count']} vectors")
    print("Connecting to existing index...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print(" Connected to existing index")
else:
    print(f" Uploading {len(text_chunks)} documents to Pinecone...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings, 
    )
    print(" Upload complete!")
import PyPDF2
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec

with open("cvc.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages])

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

client = OpenAI()
embeddings = [client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding for chunk in chunks]

# 1. Initialize pinecone client
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])  # Replace with your API key from Pinecone Console 

# 2. Create index
pc.create_index(
    name="cvc",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
index = pc.Index('cvc')

vectors = [
    {
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {"text": chunk, "document_id": "doc-123"}
    }
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
]

# Upsert in batches of 100
for i in range(0, len(vectors), 100):
    index.upsert(vectors=vectors[i:i+100])

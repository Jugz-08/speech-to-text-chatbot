import PyPDF2
import streamlit as st

with open("C:/Users/jugal/Downloads/POCs/speech_text_chatbot/doc/cvc.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = " ".join([page.extract_text() for page in reader.pages])

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_text(text)

from openai import OpenAI

client = OpenAI()
embeddings = [client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding for chunk in chunks]

from pinecone import Pinecone

# Replace with your API key from Pinecone Console   
from pinecone import Pinecone, ServerlessSpec

# 1. Initialize client
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

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

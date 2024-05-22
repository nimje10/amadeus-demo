from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pinecone 
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain_openai import OpenAIEmbeddings

import os
from pinecone import Pinecone, ServerlessSpec

directory = 'data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
print("len documents", len(documents))

def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print("len docs", len(docs))

#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Assuming you have your API key in an environment variable or replace `os.environ.get("PINECONE_API_KEY")` with your actual key string
PINECONE_API_KEY = "d203c250-37c6-4e38-8d0a-f2e4d4d4bd1b"

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "amadeus-chat"
index = pc.Index(name=index_name)

# Initialize your embeddings model
model = OpenAIEmbeddings(model_kwargs={'model_name': "ada"})

def upload_documents(docs, index, namespace, text_key='text'):
    # Transform documents to embeddings and upload to Pinecone
    for doc in docs:
        if text_key in doc:
            text = doc[text_key]
            vector = embeddings.embed([text])[0]  # Assuming embed returns a list of embeddings
            # Construct the vector item to insert: (id, vector, metadata)
            # ID could be a unique identifier from your documents, or use a hash, etc.
            vector_item = (hash(text), vector, doc)  # Including entire document as metadata
            index.upsert(vector_item)

# Example usage
upload_documents(docs, index, "PINECONE_NAMESPACE")

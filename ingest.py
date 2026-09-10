"""
ingest.py
Reads all .txt/.pdf files from sample_docs/, splits them into chunks,
embeds them, and stores them in a local Chroma vector database.

Run this once (and again any time you add new documents):
    python ingest.py
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_DIR = "sample_docs"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free


def load_documents():
    docs = []
    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if filename.endswith(".txt"):
            docs.extend(TextLoader(path).load())
        elif filename.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    return docs


def main():
    print(f"Loading documents from {DOCS_DIR}/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Building vector store at {PERSIST_DIR}/ ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print("Done. Vector store is ready.")


if __name__ == "__main__":
    main()

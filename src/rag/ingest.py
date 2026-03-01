import os
import re
from pathlib import Path
import math 

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.vectorstore.chroma_store import get_vectorstore
from src import config
from helper_function.prints import *

BATCH_SIZE = 100  # Number of chunks to process at once, useless here, but becomes usefull in case of greater dataset.

def load_documents():
    """
    Loads text files from the knowledge base directory.
    """
    data_path = config.PROJECT_ROOT / "data" / "knowledge_base"

    if not data_path.exists():
        print(red(f"⚠️ Warning: Directory {data_path} does not exist."))
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    print(f"📂 Loading documents from: {data_path}")
    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True} # prevents crashes on special characters (common in German/Euro text)
    ) # grab all .txt files

    docs = loader.load()
    print(green(f"Loaded {len(docs)} documents."))
    return docs

def clean_metadata(docs):
    """
    Clean up metadata in order that citations look nice.
    Changes absolute paths '/Users/gabriel/.../doc.txt' to just 'doc.txt'
    """
    for doc in docs:
        source_path = doc.metadata.get("source", "")
        if source_path:
            doc.metadata["source"] = os.path.basename(source_path)
    return docs

def clean_content(text: str) -> str:
    """
    Optional: Clean text artifacts.
    Replaces double newlines with single, removes excessive tabs.
    """
    # Example: "BM-\nW" -> "BMW"
    text = re.sub(r"-\n(?=\w)", "", text)

    # Collapse excessive blank lines (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text

def calculate_hnsw_params(n_chunks: int):
    """
    Adaptive HNSW Configuration.

    Logic:
    - Small N (< 1000): High params -> effectively Brute Force accuracy.
    - Large N: Standard params -> Balance speed/recall.

    return :
        - dict : parameters for HNSW
    """
    if n_chunks < 1000:
        # "Brute Force" simulation: Very high accuracy settings
        M = 64
        ef_construction = 400
        print(orange(f"Small dataset ({n_chunks} chunks): Using High Accuracy (M={M}, ef={ef_construction})"))
    else:
        # Large dataset: Standard efficient HNSW, m_L = 1 / ln(M) is handled internally by HNSW.
        M = 32
        ef_construction = 200
        print(orange(f"⚠️ Large dataset ({n_chunks} chunks): Using Balanced HNSW (M={M}, ef={ef_construction})"))

    # Return in Chroma's metadata format
    return {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": ef_construction,
        "hnsw:M": M
    }

def batch_generator(data, batch_size):
    """Yields batches of data."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]

def ingest():
    """
    Main ingestion function: Load -> Split -> Embed -> Store
    """
    raw_docs = load_documents()
    if not raw_docs:
        print(red("No documents found. Exiting."))
        return
    cleaned_docs = clean_metadata(raw_docs)

    for doc in cleaned_docs:
        doc.page_content = clean_content(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNKING_SIZE,
        chunk_overlap=config.CHUNKING_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(cleaned_docs)
    total_chunks = len(chunks)
    print(orange(f"Split into {len(chunks)} chunks (Size: {config.CHUNKING_SIZE}, Overlap: {config.CHUNKING_OVERLAP})"))

    # Init vector
    hnsw_config = calculate_hnsw_params(total_chunks)
    vectorstore = get_vectorstore(clean=True, collection_metadata=hnsw_config)

    print("⚡ Indexing in batches of {BATCH_SIZE}...")

    total_batches = math.ceil(total_chunks / BATCH_SIZE)

    for i, batch in enumerate(batch_generator(chunks, BATCH_SIZE)):
        vectorstore.add_documents(batch)
        print(f"   Batch {i+1}/{total_batches} indexed ({len(batch)} chunks)")

    print(green(f"🎉 Success! Database stored at: {config.VECTOR_DB_PATH}"))

if __name__ == "__main__":
    ingest()

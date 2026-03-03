import os
import shutil

from langchain_chroma import Chroma  

from src import config
from src.embeddings.bge_m3 import get_embedding_model

def get_vectorstore(clean: bool = False, collection_metadata: dict = None):
    """
    Returns the ChromaDB vector store instance.
    Args:
        clean (bool): If True, deletes the existing database and starts fresh. Useful for re-ingesting documents.
        collection_metadata (dict): HNSW configuration (M, ef_construction).
            Only used when creating a NEW collection.
    Return:
        langchain_chroma.vectorstores.Chroma
        
    """
    db_path = str(config.VECTOR_DB_PATH)

    if clean and os.path.exists(db_path):
        print(f"🧹 Clearing existing vector store at {db_path}...")
        shutil.rmtree(db_path)

    embedding_function = get_embedding_model()
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function,
        collection_name="bmw_knowledge_base", 
        collection_metadata=collection_metadata # HNSW with high parammeters <==> brutforce (we do it in case of short dataset)
    )
    
    return vectorstore
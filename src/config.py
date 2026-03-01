import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
MODEL_DIR = PROJECT_ROOT / "presentation" / "models"
 
GENERATION_CONFIG = {
    "max_tokens": 512,
    "temp": 0.7,
    "verbose": True
}


VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vectorstore"
CHUNKING_SIZE = 500
CHUNKING_OVERLAP = 50
top_k = 4

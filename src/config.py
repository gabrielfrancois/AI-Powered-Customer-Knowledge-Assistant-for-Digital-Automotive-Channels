import os
from pathlib import Path

# Project Structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "presentation" / "models"

# Model configuration
MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
TEMPERATURE = 0.2
GENERATION_CONFIG = {
    "max_tokens": 512,
    "temp": TEMPERATURE,
    "verbose": True # generate the answer
}

# RAG configuration
top_k = 4
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vectorstore"
CHUNKING_SIZE = 500
CHUNKING_OVERLAP = 50
BATCH_SIZE = 100    

# FEATURE FLAGS (MODULARITY)
ENABLE_ANALYTICS = True
ENABLE_FEEDBACK_COLLECTION = True

# ANALYTICS PATHS
LOGS_DIR = DATA_DIR / "logs"
INTERACTIONS_FILE = LOGS_DIR / "interactions.csv"
FEEDBACK_FILE = LOGS_DIR / "feedback.csv"

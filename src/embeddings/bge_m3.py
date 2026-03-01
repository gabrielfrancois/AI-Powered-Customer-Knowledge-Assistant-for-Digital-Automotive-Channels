from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch  

from src import config 
from helper_function.prints import *


class LocalHuggingFaceEmbeddings(Embeddings):
    """
    A custom wrapper for SentenceTransformers to be compatible with LangChain.
    We built this to avoid 'langchain-huggingface' dependency conflicts.
    """

    def __init__(self):
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu" 

        print(blue(f"Using device: {self.device}"))

        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME,
            device=self.device,
            cache_folder=str(config.MODEL_DIR) # Cache in our project folder
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        BGE-M3 works best with normalized embeddings.
        """
        # normalize_embeddings=True is crucial for Dot Product / Cosine Similarity
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        """

        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

def get_embedding_model() -> LocalHuggingFaceEmbeddings:
    return LocalHuggingFaceEmbeddings()

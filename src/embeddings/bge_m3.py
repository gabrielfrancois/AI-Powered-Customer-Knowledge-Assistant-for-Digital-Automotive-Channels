from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch  

from src import config 
from helper_function.prints import *

_STATIC_EMBEDDING_MODEL = None

class LocalHuggingFaceEmbeddings(Embeddings):
    """
    A custom wrapper for SentenceTransformers to be compatible with LangChain.
    We built this to avoid 'langchain-huggingface' dependency conflicts.
    """
    
    def __init__(self):
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        
        global _STATIC_EMBEDDING_MODEL

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu" 
        print(blue(f"Using device: {self.device}"))
            
        if _STATIC_EMBEDDING_MODEL is None:
            print(orange(f"Loading Embedding Model (BGE-M3) to {self.device}..."))
            _STATIC_EMBEDDING_MODEL = SentenceTransformer(
                config.EMBEDDING_MODEL_NAME,
                device=self.device,
                cache_folder=str(config.MODEL_DIR)
            )
            print(green("Embedding Model loaded!"))

        self.model = _STATIC_EMBEDDING_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        BGE-M3 works best with normalized embeddings.
        """
   
        safe_texts = []
        for t in texts:
            if t is None or not isinstance(t, str) or not t.strip():
                safe_texts.append(" ") 
            else:
                safe_texts.append(t.replace("\n", " "))
        embeddings = self.model.encode(safe_texts, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            text = " " 
        text = text.replace("\n", " ")
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

def get_embedding_model() -> LocalHuggingFaceEmbeddings:
    return LocalHuggingFaceEmbeddings()

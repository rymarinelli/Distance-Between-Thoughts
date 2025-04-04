from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed_steps(steps):
    """
    Encodes steps using the SentenceTransformer model.
    """
    model = get_model()
    return model.encode(steps)

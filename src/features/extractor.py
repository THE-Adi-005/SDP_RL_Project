import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

CATEGORY_MAP = {"math": 0, "logic": 1, "commonsense": 2}

def extract_features(question: str, category: str) -> np.ndarray:
    embedder = get_embedder()
    embedding = embedder.encode(question)  # 384-dim
    length = len(question.split())
    n_sentences = question.count(".") + question.count("?") + 1
    category_onehot = np.zeros(len(CATEGORY_MAP))
    category_onehot[CATEGORY_MAP.get(category, 0)] = 1.0
    engineered = np.array([length, n_sentences, *category_onehot])
    return np.concatenate([embedding, engineered]).astype(np.float32)

import numpy as np
from typing import Optional, Union
from sentence_transformers import SentenceTransformer
from src.features.multimodal_extractor import (
    extract_multimodal_features,
    compute_perceptual_features,
    get_text_embedder,
    CATEGORY_MAP
)

_embedder = None

def get_embedder():
    return get_text_embedder()

def extract_features(
    question: str,
    category: str = "general",
    image_path: Optional[str] = None,
    multimodal: bool = False
) -> np.ndarray:
    """Feature extraction with seamless multimodal support.

    If multimodal is True or image_path is provided, produces 904-dimensional
    multimodal state vector (Text 384 + Vision 512 + Perceptual 3 + Structure 2 + Category 3).
    Otherwise, returns backward-compatible 389-dimensional linguistic feature vector.
    """
    if multimodal or image_path is not None:
        return extract_multimodal_features(question=question, category=category, image_input=image_path)

    embedder = get_embedder()
    embedding = embedder.encode(question)  # 384-dim
    length = len(question.split())
    n_sentences = question.count(".") + question.count("?") + 1
    category_onehot = np.zeros(3, dtype=np.float32)
    cat_idx = CATEGORY_MAP.get(category.lower(), 0)
    category_onehot[min(cat_idx, 2)] = 1.0
    engineered = np.array([length, n_sentences, *category_onehot], dtype=np.float32)
    return np.concatenate([embedding, engineered]).astype(np.float32)

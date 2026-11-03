"""Multimodal Context Feature Extractor for CascadeRoute-VL.

Extracts joint visual-linguistic-perceptual context vector s_t ∈ ℝ^904:
- Vision Encoder: SigLIP/CLIP 512-dim visual representation
- Text Encoder: MiniLM-L6 384-dim semantic query representation
- Perceptual Features: Shannon Entropy, Sobel Edge Density, Aspect Ratio (3-dim)
- Linguistic Structural Features: Word count, sentence count (2-dim)
- Domain / Category Representation: One-hot encoded category (3-dim)
- Dynamic Resource State: Remaining budget fraction b_t ∈ [0, 1]

Reference:
- CascadeRoute-VL (CVPR 2027), Section 4.1 "Multimodal State Space".
"""
from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from sentence_transformers import SentenceTransformer

# Lazy singletons
_text_embedder: Optional[SentenceTransformer] = None
_clip_model = None
_clip_preprocess = None

CATEGORY_MAP = {
    "math": 0,
    "logic": 1,
    "commonsense": 2,
    "vqa": 1,
    "ocr": 2,
    "stem": 0
}


def get_text_embedder() -> SentenceTransformer:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _text_embedder


def compute_perceptual_features(image_input: Union[str, Path, Any]) -> np.ndarray:
    """Compute visual complexity signals: Shannon entropy, Sobel edge density, aspect ratio."""
    if not HAS_PIL or image_input is None:
        return np.zeros(3, dtype=np.float32)

    try:
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                return np.zeros(3, dtype=np.float32)
            pil_img = Image.open(p).convert("L")
            orig_img = Image.open(p)
            w, h = orig_img.size
        else:
            pil_img = image_input.convert("L")
            w, h = image_input.size

        img_arr = np.array(pil_img, dtype=np.float32)

        # 1. Shannon Entropy: information density
        hist, _ = np.histogram(img_arr.flatten(), bins=256, range=(0, 256))
        prob = hist.astype(np.float64) / max(1, img_arr.size)
        prob = prob[prob > 0]
        entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))

        # 2. Edge density via Sobel: spatial / fine-grained frequency
        if HAS_SCIPY:
            dx = ndimage.sobel(img_arr, axis=0)
            dy = ndimage.sobel(img_arr, axis=1)
            edge_density = float(np.sqrt(dx**2 + dy**2).mean() / 255.0)
        else:
            edge_density = float(np.std(img_arr) / 255.0)

        # 3. Aspect Ratio: document/chart vs standard photo
        aspect_ratio = float(w / max(1, h))

        return np.array([entropy, edge_density, aspect_ratio], dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


def extract_multimodal_features(
    question: str,
    category: str = "general",
    image_input: Optional[Union[str, Path, Any]] = None,
    precomputed_vision_emb: Optional[np.ndarray] = None,
    budget_fraction: Optional[float] = None
) -> np.ndarray:
    """Extract joint 904-dimensional multimodal state vector.

    Args:
        question: Linguistic question string.
        category: Domain category label.
        image_input: File path or PIL Image object.
        precomputed_vision_emb: Pre-extracted 512-dim visual embedding if available.
        budget_fraction: Optional b_t ∈ [0, 1] resource remaining.

    Returns:
        np.ndarray of shape (904,) or (905,) if budget is explicitly appended.
    """
    # 1. Linguistic context (384-dim)
    embedder = get_text_embedder()
    text_emb = embedder.encode(question, convert_to_numpy=True).astype(np.float32)

    # 2. Visual context (512-dim)
    if precomputed_vision_emb is not None:
        vis_emb = precomputed_vision_emb.astype(np.float32).flatten()
        if len(vis_emb) != 512:
            # Interpolate or pad to 512
            if len(vis_emb) > 512:
                vis_emb = vis_emb[:512]
            else:
                vis_emb = np.pad(vis_emb, (0, 512 - len(vis_emb)))
    else:
        # Default placeholder if no vision model loaded on CPU
        vis_emb = np.zeros(512, dtype=np.float32)

    # 3. Perceptual complexity (3-dim)
    perc = compute_perceptual_features(image_input)

    # 4. Linguistic structure (2-dim)
    words = question.split()
    word_count = float(len(words))
    n_sentences = float(question.count(".") + question.count("?") + question.count("!") + 1)

    # 5. Category one-hot (3-dim)
    cat_idx = CATEGORY_MAP.get(str(category).lower(), 1)
    cat_onehot = np.zeros(3, dtype=np.float32)
    cat_onehot[cat_idx] = 1.0

    components = [
        text_emb,                    # 384
        vis_emb,                     # 512
        perc,                        # 3
        np.array([word_count, n_sentences], dtype=np.float32), # 2
        cat_onehot                   # 3
    ]

    if budget_fraction is not None:
        components.append(np.array([float(budget_fraction)], dtype=np.float32))

    return np.concatenate(components).astype(np.float32)


import hashlib
import json
import os

CACHE_DIR = ".cache"

def _cache_key(model: str, prompt: str, params: dict) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def get_cached(model, prompt, params):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(model, prompt, params)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def set_cached(model, prompt, params, response: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(model, prompt, params)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(response, f)

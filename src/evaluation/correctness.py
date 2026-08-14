import re

def normalize_numeric(text: str) -> str:
    return re.sub(r"[^\d.\-]", "", text.replace(",", ""))

def check_correct(predicted: str, expected: str, category: str) -> bool:
    if category == "math":
        p, e = normalize_numeric(predicted), normalize_numeric(expected)
        try:
            return abs(float(p) - float(e)) < 1e-4
        except (ValueError, TypeError):
            return p == e
    if category in ("logic", "commonsense"):
        # MCQ: compare option letters, case-insensitive, first char only
        p = predicted.strip().upper()[:1]
        e = expected.strip().upper()[:1]
        return p == e
    # fallback: exact string match
    return predicted.strip().lower() == expected.strip().lower()

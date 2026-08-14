import re

def extract_final_answer(raw_text: str) -> str:
    """Pull the answer after 'Final Answer:' if present, else return the last non-empty line."""
    match = re.search(r"final answer:\s*(.+)", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""

def extract_number(text: str):
    """Extract the first numeric value from text, stripping $ and commas."""
    cleaned = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return match.group(0) if match else None

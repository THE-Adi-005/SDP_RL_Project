import re

def extract_final_answer(raw_text: str) -> str:
    """Pull the concise final answer after 'Final Answer:' or extract the final answer line/number."""
    if not raw_text:
        return ""
        
    text = raw_text.strip()
    
    # 1. Search for 'Final Answer:' prefix
    match = re.search(r"final answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        target = match.group(1).strip()
        first_line = target.split("\n")[0].strip()
        # If the line is short (under 50 chars), return it directly
        if len(first_line) <= 50:
            return first_line.rstrip(".")
            
        # If it's a long explanation paragraph containing an equation like '= 1331', extract final number
        eq_match = re.search(r"=\s*([\$€£]?\d+(?:,\d+)*(?:\.\d+)?)\.?$", first_line)
        if eq_match:
            return eq_match.group(1).replace(",", "").replace("$", "").rstrip(".")
            
        # Or look for final number at the end of the text
        num_match = re.search(r"([\$€£]?\d+(?:,\d+)*(?:\.\d+)?)\.?$", first_line)
        if num_match:
            return num_match.group(1).replace(",", "").replace("$", "").rstrip(".")
            
        return first_line.rstrip(".")

    # 2. Fallback: check last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        last_line = lines[-1].rstrip(".")
        if len(last_line) <= 50:
            return last_line
            
        # Try extracting final number from verbose line
        num_match = re.search(r"([\$€£]?\d+(?:,\d+)*(?:\.\d+)?)\.?$", last_line)
        if num_match:
            return num_match.group(1).replace(",", "").replace("$", "").rstrip(".")
            
        return last_line

    return ""

def extract_number(text: str):
    """Extract the first numeric value from text, stripping $ and commas."""
    if not text:
        return None
    cleaned = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return match.group(0) if match else None

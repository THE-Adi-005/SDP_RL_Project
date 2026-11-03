import re

def extract_numbers(text: str) -> list[float]:
    """Extract all numerical values (handling commas and currency symbols) from text."""
    if not text:
        return []
    # Strip commas e.g. 1,331 -> 1331
    cleaned = text.replace(",", "")
    # Find all valid integers and floating point numbers (e.g. 3, 1331, 1102.5, -42)
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums

def check_correct(predicted: str, expected: str, category: str) -> bool:
    if not predicted or not expected:
        return False

    predicted_clean = predicted.strip()
    expected_clean = expected.strip()

    if category == "math":
        pred_nums = extract_numbers(predicted_clean)
        exp_nums = extract_numbers(expected_clean)
        
        if not exp_nums:
            return predicted_clean.lower() == expected_clean.lower()
            
        target_num = exp_nums[0]
        
        # 1. Check if expected number matches ANY extracted number in response
        for num in pred_nums:
            if abs(num - target_num) < 1e-4:
                return True
                
        # 2. Fallback: string containing expected number/text
        expected_no_comma = expected_clean.replace(",", "")
        predicted_no_comma = predicted_clean.replace(",", "")
        return expected_no_comma in predicted_no_comma or expected_clean.lower() in predicted_clean.lower()

    if category in ("logic", "commonsense"):
        p = predicted_clean.upper()
        e = expected_clean.upper()
        
        # Check option letters (e.g., A, B, C, D)
        if len(e) == 1 and e in ("A", "B", "C", "D", "E"):
            if p.startswith(e) or re.search(rf"\b{e}\b", p):
                return True
                
        # Substring or exact match
        return e.lower() in p.lower() or p.lower() == e.lower()

    # Fallback: substring or exact string match
    return expected_clean.lower() in predicted_clean.lower() or predicted_clean.lower() == expected_clean.lower()

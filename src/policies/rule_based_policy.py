# 0=direct, 1=cot, 2=got, 3=tot
def select(state, category: str, length: int):
    if length < 15:
        return 1  # CoT for short/simple
    if category == "logic":
        return 3  # ToT for planning-heavy logic
    return 2  # GoT otherwise

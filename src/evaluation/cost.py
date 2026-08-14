# Local models are free — this returns 0, but keep the interface
# so the pipeline is unchanged if you ever swap in a paid API.
def estimate_cost(input_tokens: int, output_tokens: int, model: str = "local") -> float:
    return 0.0

from dataclasses import dataclass

@dataclass
class StrategyResult:
    predicted_answer: str
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency: float
    n_llm_calls: int
    error: str = ""

class BaseStrategy:
    name = "base"

    def __init__(self, llm_client, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template

    def solve(self, question: str) -> StrategyResult:
        raise NotImplementedError

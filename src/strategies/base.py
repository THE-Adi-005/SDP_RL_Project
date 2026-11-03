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
    """
    [SDP: STRATEGY PATTERN INTERFACE]
    This class defines the common interface for all reasoning strategies.
    By conforming to this interface, the system achieves Runtime Polymorphism.
    The RL routing engine can dynamically swap out concrete strategies (Direct, CoT, ToT) 
    at runtime without needing to know their underlying implementations.
    """
    name = "base"

    def __init__(self, llm_client, prompt_template: str):
        self.llm = llm_client
        self.prompt_template = prompt_template

    def solve(self, question: str) -> StrategyResult:
        """
        Executes the specific algorithm for this reasoning strategy.
        Concrete classes must implement this method.
        """
        raise NotImplementedError

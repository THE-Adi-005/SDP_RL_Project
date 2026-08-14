from src.strategies.base import BaseStrategy, StrategyResult
from src.llm.parser import extract_final_answer

class CoTStrategy(BaseStrategy):
    name = "cot"

    def solve(self, question: str) -> StrategyResult:
        prompt = self.prompt_template.format(question=question)
        resp = self.llm.generate(prompt)
        answer = extract_final_answer(resp.text) if not resp.error else ""
        return StrategyResult(
            predicted_answer=answer,
            raw_text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency=resp.latency,
            n_llm_calls=1,
            error=resp.error,
        )

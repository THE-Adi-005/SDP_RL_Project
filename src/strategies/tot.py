import re
from src.strategies.base import BaseStrategy, StrategyResult
from src.llm.parser import extract_final_answer

class ToTStrategy(BaseStrategy):
    name = "tot"

    def __init__(self, llm_client, prompt_template, branches=3, depth=2, eval_top_k=2):
        super().__init__(llm_client, prompt_template)
        self.branches = branches
        self.depth = depth
        self.eval_top_k = eval_top_k

    def _confidence(self, text: str) -> float:
        m = re.search(r"confidence:\s*(\d+)", text, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    def solve(self, question: str) -> StrategyResult:
        total_in, total_out, total_latency, n_calls = 0, 0, 0.0, 0
        paths = [""]

        for _ in range(self.depth):
            candidates = []
            for path in paths:
                for _ in range(self.branches):
                    prompt = self.prompt_template.format(question=question, path=path or "(none yet)")
                    resp = self.llm.generate(prompt)
                    n_calls += 1
                    total_in += resp.input_tokens
                    total_out += resp.output_tokens
                    total_latency += resp.latency
                    if resp.error:
                        continue
                    conf = self._confidence(resp.text)
                    new_path = (path + "\n" + resp.text).strip()
                    candidates.append((conf, new_path))
            candidates.sort(key=lambda x: x[0], reverse=True)
            paths = [c[1] for c in candidates[: self.eval_top_k]] or paths

        # Final answer from the best surviving path
        final_prompt = f"Question: {question}\nReasoning so far:\n{paths[0]}\n\nGive the final answer as 'Final Answer: <answer>'."
        resp = self.llm.generate(final_prompt)
        n_calls += 1
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        total_latency += resp.latency
        answer = extract_final_answer(resp.text) if not resp.error else ""

        return StrategyResult(
            predicted_answer=answer,
            raw_text=paths[0],
            input_tokens=total_in,
            output_tokens=total_out,
            latency=total_latency,
            n_llm_calls=n_calls,
            error=resp.error,
        )

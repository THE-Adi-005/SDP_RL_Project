from src.strategies.base import BaseStrategy, StrategyResult
from src.llm.parser import extract_final_answer

class GoTStrategy(BaseStrategy):
    name = "got"

    def __init__(self, llm_client, prompt_template, max_nodes=6, aggregate_top_k=2):
        super().__init__(llm_client, prompt_template)
        self.max_nodes = max_nodes
        self.aggregate_top_k = aggregate_top_k

    def solve(self, question: str) -> StrategyResult:
        total_in, total_out, total_latency, n_calls = 0, 0, 0.0, 0
        thoughts = []

        for _ in range(self.max_nodes):
            prompt = self.prompt_template.format(question=question)
            resp = self.llm.generate(prompt)
            n_calls += 1
            total_in += resp.input_tokens
            total_out += resp.output_tokens
            total_latency += resp.latency
            if not resp.error:
                thoughts.append(resp.text)

        merged = "\n---\n".join(thoughts[: self.aggregate_top_k * 2])
        final_prompt = (
            f"Question: {question}\nHere are several independent reasoning thoughts:\n{merged}\n\n"
            "Aggregate them and give the final answer as 'Final Answer: <answer>'."
        )
        resp = self.llm.generate(final_prompt)
        n_calls += 1
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        total_latency += resp.latency
        answer = extract_final_answer(resp.text) if not resp.error else ""

        return StrategyResult(
            predicted_answer=answer,
            raw_text=merged,
            input_tokens=total_in,
            output_tokens=total_out,
            latency=total_latency,
            n_llm_calls=n_calls,
            error=resp.error,
        )

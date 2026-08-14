import time
from src.evaluation.correctness import check_correct
from src.evaluation.cost import estimate_cost

def evaluate(strategy_result, expected_answer: str, category: str) -> dict:
    correct = check_correct(strategy_result.predicted_answer, expected_answer, category)
    cost = estimate_cost(strategy_result.input_tokens, strategy_result.output_tokens)
    return {
        "correct": correct,
        "predicted_answer": strategy_result.predicted_answer,
        "expected_answer": expected_answer,
        "input_tokens": strategy_result.input_tokens,
        "output_tokens": strategy_result.output_tokens,
        "total_tokens": strategy_result.input_tokens + strategy_result.output_tokens,
        "latency": strategy_result.latency,
        "estimated_cost": cost,
        "n_llm_calls": strategy_result.n_llm_calls,
        "error": strategy_result.error,
    }

import numpy as np
import yaml
from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.evaluator import evaluate
from src.features.extractor import extract_features

ACTIONS = ["direct", "cot", "got", "tot"]

class ReasoningBanditEnv:
    def __init__(self, dataframe, model_cfg_path="configs/model.yaml",
                 strat_cfg_path="configs/strategies.yaml", reward_cfg_path="configs/reward.yaml"):
        self.df = dataframe.reset_index(drop=True)
        self.idx = 0

        with open(model_cfg_path) as f:
            model_cfg = yaml.safe_load(f)
        with open(strat_cfg_path) as f:
            strat_cfg = yaml.safe_load(f)
        with open(reward_cfg_path) as f:
            self.reward_cfg = yaml.safe_load(f)

        self.llm = LLMClient(base_url=model_cfg["base_url"], model=model_cfg["model_name"],
                              temperature=model_cfg["temperature"], max_tokens=model_cfg["max_tokens"])

        def load_prompt(p):
            with open(p) as f:
                return f.read()

        self.strategies = {
            "direct": DirectStrategy(self.llm, load_prompt("prompts/direct.txt")),
            "cot": CoTStrategy(self.llm, load_prompt("prompts/cot.txt")),
            "got": GoTStrategy(self.llm, load_prompt("prompts/got.txt"), **(strat_cfg.get("got") or {})),
            "tot": ToTStrategy(self.llm, load_prompt("prompts/tot.txt"), **(strat_cfg.get("tot") or {})),
        }
        self.max_tokens_seen = 1000  # running normalizer, updated as we go
        self.max_latency_seen = 10.0

    def reset(self):
        row = self.df.iloc[self.idx]
        state = extract_features(row["question"], row["category"])
        return state, row

    def step(self, action_idx: int, row):
        strat_name = ACTIONS[action_idx]
        strategy = self.strategies[strat_name]
        result = strategy.solve(row["question"])
        metrics = evaluate(result, str(row["expected_answer"]), row["category"])

        self.max_tokens_seen = max(self.max_tokens_seen, metrics["total_tokens"])
        self.max_latency_seen = max(self.max_latency_seen, metrics["latency"])
        norm_tokens = metrics["total_tokens"] / self.max_tokens_seen
        norm_latency = metrics["latency"] / self.max_latency_seen

        alpha, beta, gamma = self.reward_cfg["alpha"], self.reward_cfg["beta"], self.reward_cfg["gamma"]
        reward = alpha * float(metrics["correct"]) - beta * norm_tokens - gamma * norm_latency

        metrics.update({"strategy": strat_name, "reward": reward,
                         "problem_id": row["problem_id"], "category": row["category"]})

        self.idx += 1
        done = self.idx >= len(self.df)
        return reward, metrics, done

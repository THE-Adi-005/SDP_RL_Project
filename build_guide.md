# Adaptive LLM Reasoning — Complete Build Guide

Step-by-step, command-by-command instructions to build this project locally using Ollama (no paid API keys). Follow the phases in order — do not skip ahead.

**Target hardware assumed:** 6GB VRAM / 24GB RAM (adjust model sizes if yours differs — see Phase 2).

---

## Phase 0 — Prerequisites

Check these before starting:

```bash
python3 --version        # need 3.10 or 3.11
git --version
nvidia-smi                # confirm GPU + driver visible (optional but recommended)
```

If Python is missing or too old (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip -y
```

macOS (Homebrew):
```bash
brew install python@3.11 git
```

Windows: install Python 3.11 from python.org (check "Add to PATH" during install) and Git from git-scm.com.

---

## Phase 1 — Install Ollama and Pull Models

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download
```

Start the Ollama service (it usually auto-starts; if not):
```bash
ollama serve &
```

Pull the models sized for 6GB VRAM:
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M    # primary — reasoning quality
ollama pull qwen2.5:3b-instruct-q4_K_M    # fast fallback for bandit training

# optional, only for a side experiment — NOT the core benchmark:
# ollama pull deepseek-r1:7b
```

Verify it works:
```bash
ollama run qwen2.5:7b-instruct-q4_K_M "What is 12 * 8? Answer with just the number."
```

Confirm the OpenAI-compatible endpoint is up:
```bash
curl http://localhost:11434/v1/models
```

---

## Phase 2 — Repository and Environment Setup

```bash
mkdir adaptive-llm-reasoning && cd adaptive-llm-reasoning
git init

python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

python -m pip install --upgrade pip
```

Create `requirements.txt`:

```bash
cat > requirements.txt << 'EOF'
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
pydantic==2.7.4
python-dotenv==1.0.1
tqdm==4.66.4
matplotlib==3.9.0
seaborn==0.13.2
gymnasium==0.29.1
stable-baselines3==2.3.2
torch==2.3.1
sentence-transformers==3.0.1
datasets==2.20.0
openai==1.35.10
pyyaml==6.0.1
scipy==1.13.1
EOF

pip install -r requirements.txt
```

Create `.env.example` and `.gitignore`:

```bash
cat > .env.example << 'EOF'
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
MODEL_NAME=qwen2.5:7b-instruct-q4_K_M
EOF

cp .env.example .env

cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
.env
data/raw/
data/processed/
data/results/
.cache/
*.ipynb_checkpoints
EOF
```

---

## Phase 3 — Directory Structure

```bash
mkdir -p configs data/raw data/processed data/results prompts \
  src/llm src/strategies src/evaluation src/features src/rl src/policies src/utils \
  experiments/baseline experiments/bandit experiments/rl experiments/ablation \
  notebooks reports/figures reports/tables tests

touch src/__init__.py src/llm/__init__.py src/strategies/__init__.py \
  src/evaluation/__init__.py src/features/__init__.py src/rl/__init__.py \
  src/policies/__init__.py src/utils/__init__.py
```

Resulting structure:

```text
adaptive-llm-reasoning/
├── README.md
├── requirements.txt
├── .env / .env.example
├── .gitignore
├── configs/
│   ├── model.yaml
│   ├── datasets.yaml
│   ├── strategies.yaml
│   ├── reward.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── prompts/
│   ├── direct.txt
│   ├── cot.txt
│   ├── got.txt
│   └── tot.txt
├── src/
│   ├── llm/{client.py, parser.py}
│   ├── strategies/{base.py, direct.py, cot.py, got.py, tot.py}
│   ├── evaluation/{correctness.py, cost.py, latency.py, evaluator.py}
│   ├── features/extractor.py
│   ├── rl/{environment.py, bandit.py, dqn.py, ppo.py}
│   ├── policies/{random_policy.py, always_x_policy.py, rule_based_policy.py}
│   └── utils/{caching.py, logging.py, io.py}
├── run_baseline.py
├── run_bandit.py
├── run_rl.py
├── experiments/{baseline,bandit,rl,ablation}/
├── notebooks/
├── reports/{figures,tables}/
└── tests/
```

---

## Phase 4 — Config Files

```bash
cat > configs/model.yaml << 'EOF'
provider: ollama
base_url: http://localhost:11434/v1
model_name: qwen2.5:7b-instruct-q4_K_M
fallback_model: qwen2.5:3b-instruct-q4_K_M
temperature: 0.2
max_tokens: 512
timeout_seconds: 60
EOF

cat > configs/strategies.yaml << 'EOF'
direct: {}
cot: {}
got:
  max_nodes: 6
  aggregate_top_k: 2
tot:
  branches: 3
  depth: 2
  eval_top_k: 2
EOF

cat > configs/reward.yaml << 'EOF'
alpha: 1.0     # correctness weight
beta: 0.3      # token cost weight
gamma: 0.1     # latency weight
EOF

cat > configs/datasets.yaml << 'EOF'
gsm8k:
  path: data/processed/gsm8k.csv
  category: math
logiqa:
  path: data/processed/logiqa.csv
  category: logic
commonsenseqa:
  path: data/processed/commonsenseqa.csv
  category: commonsense
split:
  train: 0.7
  val: 0.15
  test: 0.15
seed: 42
EOF

cat > configs/experiment.yaml << 'EOF'
run_name: baseline_v1
output_dir: data/results
seed: 42
EOF
```

---

## Phase 5 — Download and Prepare Datasets

Install/confirm the `datasets` library (already in requirements.txt), then pull each benchmark.

```bash
mkdir -p data/raw
```

Create `scripts_download.py` (a one-off script — run then discard, or keep in `src/utils/`):

```bash
cat > download_datasets.py << 'EOF'
"""Download GSM8K, LogiQA, CommonsenseQA and write unified CSVs to data/processed/."""
import pandas as pd
from datasets import load_dataset

# --- GSM8K (math) ---
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
rows = []
for i, ex in enumerate(gsm8k):
    answer = ex["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({
        "problem_id": f"gsm8k_{i}",
        "question": ex["question"],
        "expected_answer": answer,
        "category": "math",
        "difficulty": "unknown",
    })
pd.DataFrame(rows).to_csv("data/processed/gsm8k.csv", index=False)
print(f"GSM8K: {len(rows)} problems saved")

# --- LogiQA (logic, MCQ) ---
logiqa = load_dataset("lucasmccabe/logiqa", split="test")
rows = []
for i, ex in enumerate(logiqa):
    options = ex["options"]
    correct_idx = ex["correct_option"]
    letters = ["A", "B", "C", "D"]
    formatted_q = ex["query"] + " " + ex["context"] + "\n" + \
        "\n".join(f"{letters[j]}. {opt}" for j, opt in enumerate(options))
    rows.append({
        "problem_id": f"logiqa_{i}",
        "question": formatted_q,
        "expected_answer": letters[correct_idx],
        "category": "logic",
        "difficulty": "unknown",
    })
pd.DataFrame(rows).to_csv("data/processed/logiqa.csv", index=False)
print(f"LogiQA: {len(rows)} problems saved")

# --- CommonsenseQA ---
csqa = load_dataset("tau/commonsense_qa", split="validation")
rows = []
for i, ex in enumerate(csqa):
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    formatted_q = ex["question"] + "\n" + \
        "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    rows.append({
        "problem_id": f"csqa_{i}",
        "question": formatted_q,
        "expected_answer": ex["answerKey"],
        "category": "commonsense",
        "difficulty": "unknown",
    })
pd.DataFrame(rows).to_csv("data/processed/commonsenseqa.csv", index=False)
print(f"CommonsenseQA: {len(rows)} problems saved")
EOF

python download_datasets.py
```

Sample down to a manageable size for a course project (e.g. 100 per category to start, scale later):

```bash
cat > sample_datasets.py << 'EOF'
import pandas as pd

for name in ["gsm8k", "logiqa", "commonsenseqa"]:
    df = pd.read_csv(f"data/processed/{name}.csv")
    sample = df.sample(n=min(150, len(df)), random_state=42)
    sample.to_csv(f"data/processed/{name}_sample.csv", index=False)
    print(f"{name}: sampled {len(sample)} / {len(df)}")
EOF

python sample_datasets.py
```

---

## Phase 6 — Prompt Templates

```bash
cat > prompts/direct.txt << 'EOF'
Answer the following question directly with only the final answer, no explanation.

Question: {question}

Final Answer:
EOF

cat > prompts/cot.txt << 'EOF'
Solve the following question step by step, showing your reasoning, then give the final answer on the last line in the exact format "Final Answer: <answer>".

Question: {question}

Let's think step by step.
EOF

cat > prompts/got.txt << 'EOF'
You are solving a problem using multiple reasoning branches that will later be merged.

Question: {question}

Generate one independent reasoning thought that approaches the problem from a specific angle. End with "Thought: <your reasoning>" on its own line.
EOF

cat > prompts/tot.txt << 'EOF'
You are exploring a solution tree for the following problem.

Question: {question}
Current partial reasoning path: {path}

Propose the next reasoning step, then rate your own confidence in this path from 1-10.
Format:
Step: <next reasoning step>
Confidence: <1-10>
EOF
```

---

## Phase 7 — LLM Client (`src/llm/client.py`)

```bash
cat > src/llm/client.py << 'EOF'
import os
import time
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    error: str = ""

class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None,
                 temperature=0.2, max_tokens=512, timeout=60):
        self.client = OpenAI(
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=api_key or os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        self.model = model or os.getenv("MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            latency = time.time() - start
            text = resp.choices[0].message.content
            usage = resp.usage
            return LLMResponse(
                text=text,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency=latency,
            )
        except Exception as e:
            return LLMResponse(text="", input_tokens=0, output_tokens=0,
                                latency=time.time() - start, error=str(e))
EOF
```

Answer parser:

```bash
cat > src/llm/parser.py << 'EOF'
import re

def extract_final_answer(raw_text: str) -> str:
    """Pull the answer after 'Final Answer:' if present, else return the last non-empty line."""
    match = re.search(r"final answer:\s*(.+)", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().split("\n")[0].strip()
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""

def extract_number(text: str):
    """Extract the first numeric value from text, stripping $ and commas."""
    cleaned = text.replace(",", "").replace("$", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    return match.group(0) if match else None
EOF
```

---

## Phase 8 — Caching Layer (`src/utils/caching.py`)

```bash
cat > src/utils/caching.py << 'EOF'
import hashlib
import json
import os

CACHE_DIR = ".cache"

def _cache_key(model: str, prompt: str, params: dict) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def get_cached(model, prompt, params):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(model, prompt, params)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def set_cached(model, prompt, params, response: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(model, prompt, params)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(response, f)
EOF
```

---

## Phase 9 — Strategy Base Class + Direct (`src/strategies/`)

```bash
cat > src/strategies/base.py << 'EOF'
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
EOF

cat > src/strategies/direct.py << 'EOF'
from src.strategies.base import BaseStrategy, StrategyResult
from src.llm.parser import extract_final_answer

class DirectStrategy(BaseStrategy):
    name = "direct"

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
EOF

cat > src/strategies/cot.py << 'EOF'
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
EOF
```

`src/strategies/tot.py` (capped branching for 6GB VRAM — keep `branches=3, depth=2`):

```bash
cat > src/strategies/tot.py << 'EOF'
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
EOF
```

`src/strategies/got.py` (simplified: generate N independent thoughts, aggregate):

```bash
cat > src/strategies/got.py << 'EOF'
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
EOF
```

---

## Phase 10 — Evaluation Modules (`src/evaluation/`)

```bash
cat > src/evaluation/correctness.py << 'EOF'
import re

def normalize_numeric(text: str) -> str:
    return re.sub(r"[^\d.\-]", "", text.replace(",", ""))

def check_correct(predicted: str, expected: str, category: str) -> bool:
    if category == "math":
        p, e = normalize_numeric(predicted), normalize_numeric(expected)
        try:
            return abs(float(p) - float(e)) < 1e-4
        except (ValueError, TypeError):
            return p == e
    if category in ("logic", "commonsense"):
        # MCQ: compare option letters, case-insensitive, first char only
        p = predicted.strip().upper()[:1]
        e = expected.strip().upper()[:1]
        return p == e
    # fallback: exact string match
    return predicted.strip().lower() == expected.strip().lower()
EOF

cat > src/evaluation/cost.py << 'EOF'
# Local models are free — this returns 0, but keep the interface
# so the pipeline is unchanged if you ever swap in a paid API.
def estimate_cost(input_tokens: int, output_tokens: int, model: str = "local") -> float:
    return 0.0
EOF

cat > src/evaluation/evaluator.py << 'EOF'
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
EOF
```

---

## Phase 11 — Baseline Experiment Runner (`run_baseline.py`)

```bash
cat > run_baseline.py << 'EOF'
import argparse
import os
import yaml
import pandas as pd
from tqdm import tqdm
from src.llm.client import LLMClient
from src.strategies.direct import DirectStrategy
from src.strategies.cot import CoTStrategy
from src.strategies.got import GoTStrategy
from src.strategies.tot import ToTStrategy
from src.evaluation.evaluator import evaluate

STRATEGY_MAP = {
    "direct": (DirectStrategy, "prompts/direct.txt"),
    "cot": (CoTStrategy, "prompts/cot.txt"),
    "got": (GoTStrategy, "prompts/got.txt"),
    "tot": (ToTStrategy, "prompts/tot.txt"),
}

def load_prompt(path):
    with open(path) as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. gsm8k_sample")
    parser.add_argument("--strategies", required=True, help="comma-separated: direct,cot,got,tot")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/strategies.yaml") as f:
        strat_cfg = yaml.safe_load(f)

    llm = LLMClient(
        base_url=model_cfg["base_url"],
        model=model_cfg["model_name"],
        temperature=model_cfg["temperature"],
        max_tokens=model_cfg["max_tokens"],
        timeout=model_cfg["timeout_seconds"],
    )

    df = pd.read_csv(f"data/processed/{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)

    results = []
    strategies = args.strategies.split(",")
    for strat_name in strategies:
        cls, prompt_path = STRATEGY_MAP[strat_name]
        prompt_template = load_prompt(prompt_path)
        kwargs = strat_cfg.get(strat_name, {}) or {}
        strategy = cls(llm, prompt_template, **kwargs)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=strat_name):
            sr = strategy.solve(row["question"])
            metrics = evaluate(sr, str(row["expected_answer"]), row["category"])
            metrics.update({
                "problem_id": row["problem_id"],
                "category": row["category"],
                "difficulty": row.get("difficulty", "unknown"),
                "strategy": strat_name,
                "model": model_cfg["model_name"],
            })
            results.append(metrics)

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{args.dataset}_baseline_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} rows to {out_path}")

if __name__ == "__main__":
    main()
EOF
```

**Run the first small test (50-100 problems, per your project plan §15):**

```bash
python run_baseline.py --dataset gsm8k_sample --strategies direct,cot --limit 20
```

Inspect manually:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/results/gsm8k_sample_baseline_results.csv'); print(df[['strategy','correct','total_tokens','latency']].groupby('strategy').mean(numeric_only=True))"
```

Once direct/cot look correct, add got/tot and scale up:
```bash
python run_baseline.py --dataset gsm8k_sample --strategies direct,cot,got,tot
python run_baseline.py --dataset logiqa_sample --strategies direct,cot,got,tot
python run_baseline.py --dataset commonsenseqa_sample --strategies direct,cot,got,tot
```

---

## Phase 12 — Feature Extractor (`src/features/extractor.py`)

```bash
cat > src/features/extractor.py << 'EOF'
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

CATEGORY_MAP = {"math": 0, "logic": 1, "commonsense": 2}

def extract_features(question: str, category: str) -> np.ndarray:
    embedder = get_embedder()
    embedding = embedder.encode(question)  # 384-dim
    length = len(question.split())
    n_sentences = question.count(".") + question.count("?") + 1
    category_onehot = np.zeros(len(CATEGORY_MAP))
    category_onehot[CATEGORY_MAP.get(category, 0)] = 1.0
    engineered = np.array([length, n_sentences, *category_onehot])
    return np.concatenate([embedding, engineered]).astype(np.float32)
EOF
```

First run downloads the embedding model automatically:
```bash
python -c "from src.features.extractor import extract_features; print(extract_features('What is 2+2?', 'math').shape)"
```

---

## Phase 13 — Bandit (`src/rl/bandit.py`)

```bash
cat > src/rl/bandit.py << 'EOF'
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions=4, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05, lr=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.q_values = {}  # action -> running average reward (context-free fallback)
        self.weights = None  # linear model per action, set on first update

    def _init_weights(self, dim):
        self.weights = np.zeros((self.n_actions, dim))

    def select_action(self, state: np.ndarray) -> int:
        if self.weights is None:
            self._init_weights(len(state))
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        scores = self.weights @ state
        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        if self.weights is None:
            self._init_weights(len(state))
        prediction = self.weights[action] @ state
        error = reward - prediction
        self.weights[action] += self.lr * error * state / (np.linalg.norm(state) + 1e-8)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
EOF
```

---

## Phase 14 — RL Environment (`src/rl/environment.py`)

```bash
cat > src/rl/environment.py << 'EOF'
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
EOF
```

---

## Phase 15 — Bandit Runner (`run_bandit.py`)

```bash
cat > run_bandit.py << 'EOF'
import argparse
import os
import pandas as pd
from tqdm import tqdm
from src.rl.environment import ReasoningBanditEnv, ACTIONS
from src.rl.bandit import EpsilonGreedyBandit
from src.features.extractor import extract_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(f"data/processed/{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)

    env = ReasoningBanditEnv(df)
    bandit = EpsilonGreedyBandit(n_actions=len(ACTIONS))

    logs = []
    cumulative_reward = 0.0
    for _ in tqdm(range(len(df)), desc="bandit training"):
        state, row = env.reset()
        action = bandit.select_action(state)
        reward, metrics, done = env.step(action, row)
        bandit.update(state, action, reward)

        cumulative_reward += reward
        metrics["cumulative_reward"] = cumulative_reward
        metrics["epsilon"] = bandit.epsilon
        logs.append(metrics)
        if done:
            break

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{args.dataset}_bandit_results.csv"
    pd.DataFrame(logs).to_csv(out_path, index=False)
    print(f"Saved {len(logs)} rows to {out_path}")

if __name__ == "__main__":
    main()
EOF
```

Run it:
```bash
python run_bandit.py --dataset gsm8k_sample --limit 100
```

---

## Phase 16 — Non-RL Baseline Policies (`src/policies/`)

```bash
cat > src/policies/random_policy.py << 'EOF'
import random
def select(state, n_actions=4):
    return random.randint(0, n_actions - 1)
EOF

cat > src/policies/always_x_policy.py << 'EOF'
def make_always_policy(action_idx: int):
    def select(state):
        return action_idx
    return select
EOF

cat > src/policies/rule_based_policy.py << 'EOF'
# 0=direct, 1=cot, 2=got, 3=tot
def select(state, category: str, length: int):
    if length < 15:
        return 1  # CoT for short/simple
    if category == "logic":
        return 3  # ToT for planning-heavy logic
    return 2  # GoT otherwise
EOF
```

---

## Phase 17 — Plotting / Analysis Script

```bash
cat > analyze_results.py << 'EOF'
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    os.makedirs("reports/figures", exist_ok=True)

    summary = df.groupby("strategy").agg(
        accuracy=("correct", "mean"),
        avg_tokens=("total_tokens", "mean"),
        median_latency=("latency", "median"),
    )
    print(summary)
    summary.to_csv("reports/tables/summary.csv")

    fig, ax = plt.subplots()
    summary["accuracy"].plot(kind="bar", ax=ax, title="Accuracy by Strategy")
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy_by_strategy.png")

    fig, ax = plt.subplots()
    ax.scatter(summary["avg_tokens"], summary["accuracy"])
    for name, row in summary.iterrows():
        ax.annotate(name, (row["avg_tokens"], row["accuracy"]))
    ax.set_xlabel("Avg Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Token Cost")
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy_vs_cost.png")

    if "cumulative_reward" in df.columns:
        fig, ax = plt.subplots()
        df["cumulative_reward"].plot(ax=ax, title="Cumulative Reward Over Time")
        plt.tight_layout()
        plt.savefig("reports/figures/cumulative_reward.png")

    print("Saved figures to reports/figures/")

if __name__ == "__main__":
    main()
EOF

python analyze_results.py --results data/results/gsm8k_sample_baseline_results.csv
```

---

## Phase 18 — Basic Tests (`tests/`)

```bash
cat > tests/test_correctness.py << 'EOF'
from src.evaluation.correctness import check_correct

def test_math_correct():
    assert check_correct("72", "72", "math") is True

def test_math_incorrect():
    assert check_correct("70", "72", "math") is False

def test_mcq_correct():
    assert check_correct("A", "A", "commonsense") is True

def test_mcq_case_insensitive():
    assert check_correct("a", "A", "logic") is True
EOF

cat >> requirements.txt << 'EOF'
pytest==8.2.2
EOF
pip install pytest

pytest tests/ -v
```

---

## Phase 19 — Full Run Order (summary)

```bash
# 1. small pilot per strategy per dataset
python run_baseline.py --dataset gsm8k_sample --strategies direct,cot --limit 20

# 2. full baseline sweep once pilot looks correct
python run_baseline.py --dataset gsm8k_sample --strategies direct,cot,got,tot
python run_baseline.py --dataset logiqa_sample --strategies direct,cot,got,tot
python run_baseline.py --dataset commonsenseqa_sample --strategies direct,cot,got,tot

# 3. analyze baseline
python analyze_results.py --results data/results/gsm8k_sample_baseline_results.csv

# 4. bandit training
python run_bandit.py --dataset gsm8k_sample

# 5. analyze bandit run (accuracy + cumulative reward)
python analyze_results.py --results data/results/gsm8k_sample_bandit_results.csv

# 6. tests
pytest tests/ -v
```

---

## Phase 20 — Git Commit Checkpoints

Commit at each working milestone, not just at the end:

```bash
git add .
git commit -m "Phase 0-4: env, config, directory structure"

git add .
git commit -m "Phase 5-6: datasets downloaded, prompts written"

git add .
git commit -m "Phase 7-11: LLM client, strategies, baseline runner working"

git add .
git commit -m "Phase 12-16: features, bandit, RL environment working"

git add .
git commit -m "Phase 17-18: plotting, tests"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on `localhost:11434` | Ollama not running | `ollama serve &` |
| Very slow ToT runs | Branching too wide for 6GB VRAM | Lower `branches`/`depth` in `configs/strategies.yaml` |
| `CUDA out of memory` | Model too large for VRAM | Switch to `qwen2.5:3b-instruct-q4_K_M` in `configs/model.yaml` |
| Empty `predicted_answer` | Parser not matching model output format | Check `raw_text` column in results.csv, adjust prompt to force `Final Answer:` format |
| Bandit reward stuck at 0 | Reward normalizer never updated | Confirm `max_tokens_seen`/`max_latency_seen` are updating in `environment.py` |

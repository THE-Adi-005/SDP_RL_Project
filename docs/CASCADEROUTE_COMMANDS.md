# CascadeRoute: Step-by-Step Setup and Execution Guide

This document contains the commands to run the novel `CascadeRoute` architecture, which implements the 3 novel contributions for the ICLR 2027 paper:
1. Joint Model x Strategy Bandit
2. Confidence-Gated Cascade
3. Dynamic Budget Tracking

## 1. Prerequisites (Ollama Models)
Make sure the Ollama application is running. CascadeRoute routes between a strong model and a weak model, so you need both:
```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_K_M
```

## 2. Environment Setup
Activate the virtual environment. If you haven't set it up yet, create it and install dependencies:
```powershell
# Create the virtual environment (Run Once)
python -m venv venv1

# Activate and install dependencies (Run Once)
.\venv1\Scripts\activate
pip install -r requirements.txt
```

**Daily Usage (Run Every Time):**
```powershell
.\venv1\Scripts\activate
```

## 3. Data Preparation
If you haven't downloaded the datasets yet, run:
```powershell
# Download the raw datasets
python download_datasets.py

# Create small samples for testing
python sample_datasets.py
```

## 4. Running CascadeRoute
Run the new CascadeRoute experiment runner. You can specify the dataset, the total token budget, and a limit on the number of questions to process.

```powershell
# Run on the sampled math dataset with a budget of 10,000 tokens
python -m src.cascaderoute.run_cascaderoute --datasets gsm8k_sample --budget 10000 --limit 20
```

*What happens when you run this?*
- The LinUCB bandit will select an initial action (e.g., `3b_cot`).
- The `ConfidenceGate` will run the query twice. If they match, it accepts. If not, it escalates (e.g., to `7b_cot`).
- The token usage is deducted from the 10,000 budget.
- As the budget depletes, the bandit learns to prefer cheaper actions.

## 5. Analyzing Results
After the run finishes, analyze the results to see the escalations and budget impact:
```powershell
python analyze_results.py --results data/results/cascaderoute_results.csv
```
This will output tables and figures in the `reports/` folder.

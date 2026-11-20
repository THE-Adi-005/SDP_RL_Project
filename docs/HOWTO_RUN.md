# CascadeRoute: End-to-End Setup & Execution Guide

## 🚀 Daily Quick Start (If already set up)
If you have already installed the dependencies and generated the results, you only need to run this command to boot the dashboard:
```powershell
cd dashboard
python -m uvicorn app:app --reload --port 8000
```
*(Ensure the Ollama app is running in the background).*
---

## 🎯 Project Scope: SDP vs RL
This repository serves a dual academic purpose. Work should be evaluated based on the following tracks:
1. **Software Development Project (SDP Track):** Focuses on the architectural design, distributed systems scaling (Ray), the robust Confidence-Gated fallback mechanics, and the final Springer Nature Journal publication located in the `SDP_Journal/` directory.
2. **Reinforcement Learning (RL Track):** Focuses on the Contextual Bandit intelligence (`linucb.py`), the theoretical RL environment design, and the applied Application/UI Demo located in the `dashboard/` directory.

---

This guide below provides the exact commands required to set up the environment from a completely blank slate.

## 1. Environment Setup

**1.1 Install Python Dependencies**
Ensure you are using Python 3.10+ and have created a virtual environment (e.g., `venv3135`).
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install dashboard requirements explicitly if needed
pip install fastapi uvicorn
```

**1.2 Setup Ollama & Local Models**
You must have [Ollama](https://ollama.com/) installed and running on your host machine(s). Pull the required Qwen models:
```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b-instruct-q4_K_M
```

## 2. Distributed Inference via Ray

Because running 24,000+ LLM inferences on local GPUs takes time, this project uses Ray to distribute the workload.

**2.1 Start the Ray Head Node (On your main machine)**
```powershell
ray start --head --port=6379
```
*(Note the IP address output by this command, e.g., `172.19.140.108:6379`)*

**2.2 Start Ray Worker Nodes (On any secondary GPU machines)**
```powershell
ray start --address='<HEAD_NODE_IP>:6379'
```

## 3. Data Preparation

Download and prepare the GSM8K, LogiQA, and CommonsenseQA datasets.
```powershell
python download_datasets.py
```
*(This will populate the `data/` folder with the necessary JSONL files).*

## 4. Run the Evaluations

**4.1 Run the Static Baselines (Ground Truth)**
This command evaluates all 6 static models/strategies (`3b_direct`, `3b_cot`, `7b_direct`, `7b_cot`, `7b_got`, `7b_tot`). It uses the Ray cluster to spawn multiple actors.
```powershell
python -m src.evaluation.run_full_baselines --datasets gsm8k,logiqa,commonsenseqa --use-ray --num-actors 8
```
*Output: `data/results/full_baselines_results.csv`*

**4.2 Run the CascadeRoute RL Agent**
This runs the LinUCB Contextual Bandit, dynamically routing queries based on the budget constraint and utilizing the Confidence Gate for safety.
```powershell
python -m src.cascaderoute.run_cascaderoute
```
*Output: `data/results/cascaderoute_results.csv`*

## 5. Generate Academic Visualizations

Generate the Pareto Frontier and Routing Distribution charts for the ICLR paper.
```powershell
python generate_iclr_visuals.py
```
*(Images will be saved to `reports/figures/` and can be copied to your LaTeX directory).*

## 6. Launch the Dashboard

Start the FastAPI dashboard to visualize the dynamic routing, accuracy metrics, and token cost savings interactively.
```powershell
cd dashboard
python -m uvicorn app:app --reload --port 8000
```
Then, open your web browser and navigate to: **http://127.0.0.1:8000**

<div align="center">

<p align="center">
  <img src="assets/icon.png" width="220" alt="VL-RouterBench logo" />
</p>

### VL-RouterBench: A Benchmark for Vision–Language Model Routing

[![arXiv](https://img.shields.io/badge/arXiv-2512.23562-red.svg)](https://arxiv.org/abs/2512.23562)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/K1nght/VL-RouterBench)

</div>

## Overview

We provides a clean, reproducible implementation of **VL-RouterBench**, a benchmark and toolkit for **routing across a pool of Vision–Language Models (VLMs)** under both **performance** and **performance–cost** objectives.

<p align="center">
  <img src="assets/pipeline.png" width="900" alt="VL-RouterBench pipeline" />
</p>

## 📦 Data Preparation

VL-RouterBench converts [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit) outputs into a unified routing benchmark.

To make data setup easier, we provide a pre-packaged archive **`vlm_router_data.tar.gz`** that contains everything needed to run the pipeline. You can download it from any of the following channels and extract it under the repo root:

- **Google Drive**: [vlm_router_data.tar.gz](https://drive.google.com/file/d/1Va18MW8nJqvatxDXQDQq0t9NAqr93hMg/view?usp=sharing)
- **Baidu Netdisk**: [vlm_router_data.tar.gz](https://pan.baidu.com/s/1D_P8YwY_E5kDA5dUB-ovng) (code: xb1s)
- **Hugging Face**: [vlm_router_data.tar.gz](https://huggingface.co/datasets/KinghtH/VL-RouterBench)

After downloading, extract it as:

```bash
tar -xzf vlm_router_data.tar.gz
```

By default, the pipeline expects the following directories (relative to repo root):

```bash
vlm_router_data/
  VLMEvalKit_evaluation/   # required (for is_correct / evaluation)
  VLMEvalKit_inference/    # required for accurate output-token counting (Step 2)
  TSV_images/              # optional (for TSV-packed image datasets)
```

Notes:
- **`VLMEvalKit_evaluation/`** is used by Step 1 & 4 (contains correctness signals).
- **`VLMEvalKit_inference/`** is used by Step 2 (extract real model outputs to count output tokens).
- **`TSV_images/`** is used by routers for training and inference to make routing decisions.

## 📝 Citation

If you find this benchmark useful, please cite:

```bibtex
@misc{huang2025vlrouterbenchbenchmarkvisionlanguagemodel,
      title={VL-RouterBench: A Benchmark for Vision-Language Model Routing}, 
      author={Zhehao Huang and Baijiong Lin and Jingyuan Zhang and Jingying Wang and Yuhang Liu and Ning Lu and Tao Li and Xiaolin Huang},
      year={2025},
      eprint={2512.23562},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.23562}, 
}
```

---
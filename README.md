# ILVR: Interleaved Latent Visual Reasoning with Selective Perceptual Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2512.05665-b31b1b.svg)](https://arxiv.org/abs/2512.05665)
[![Dataset](https://img.shields.io/badge/Dataset-CoMT-yellow)](https://huggingface.co/datasets/shuai22/comt)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Authors:** Shuai Dong, Siyuan Wang, Xingyu Liu, Zhongyu Wei

**Affiliations:** China University of Geosciences, Wuhan; Shanghai Innovation Institute; University of Southern California; Fudan University.

---

## 📖 Abstract

<!-- Please ensure you have created an 'assets' folder and uploaded 'framework.png' -->
![Model Architecture](assets/framework.pdf)

> **Abstract:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of repeatedly re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet currently forces a critical trade-off: methods either sacrifice precise perceptual modeling by over-compressing features or fail to model dynamic problems due to static, non-interleaved structures. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. To enable this, we employ a self-supervision strategy where a Momentum Teacher Model selectively distills relevant features from \textit{helper images} into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR significantly outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning.

---

## 📢 News
* **[2025-12-08]** The code is released.
* **[2025-12-08]** The paper is released on [arXiv](https://arxiv.org/abs/2512.05665).

---

## 🛠️ Installation

### 1. Environment Setup
The code is tested with **Python 3.11**. We recommend using Conda for environment management.

```bash
# 1. Create a conda environment
conda create -n ilvr python=3.11
conda activate ilvr

# 2. Install standard dependencies
pip install -r requirements.txt

# 3. Install custom Transformers library
# ILVR requires modifications to the standard transformers library.
# We provide the modified source code in this repository.
cd transformers
pip install -e .
cd ..
```

### 2. Accelerate Configuration
This project uses HuggingFace `accelerate` for distributed training. Please configure it before running the training script.

```bash
accelerate config
```

---

## 📚 Data Preparation

We utilize the **CoMT** dataset (Chain of Multi-modal Thought). For more details about the benchmark, please refer to the [CoMT paper](https://arxiv.org/abs/2412.12932).

### Download Data
We provide the processed data on HuggingFace. Please download it from [shuai22/comt](https://huggingface.co/datasets/shuai22/comt) and organize the directory as follows.

1. Download `TRAIN.jsonl`, `TEST.jsonl`, and `comt.tar.gz`.
2. Extract the images from the tarball.

**Expected Directory Structure:**
```text
ILVR/
├── data/
│   ├── TRAIN.jsonl
│   ├── TEST.jsonl
│   └── images_comt/      <-- Extracted from comt.tar.gz
│       ├── creation/
│       └── ...
├── src/
├── transformers/
├── run_training.sh
└── README.md
```

### Data Format
The dataset follows the JSONL format:
- **text_input**: The question/instruction.
- **image_input**: Initial input images.
- **sequence_plan**: The interleaved chain-of-thought rationale containing "text", "latent" codes, and "helper_image" paths.

---

## 🚀 Training

We provide a shell script `run_training.sh` to launch distributed training.

### 1. Configure Script
Open `run_training.sh` and modify the paths to match your local setup:

```bash
# In run_training.sh:

# Path to the directory containing TRAIN.jsonl
DATA_PATH="/path/to/your/data" 

# Directory to save model checkpoints
SAVE_MODEL_PATH="/path/to/save/checkpoints"

# File path for training logs
LOG_FILE="/path/to/save/train.log"

# (Optional) HuggingFace Cache Directory
export HF_HOME="/path/to/cache" 
```

### 2. Run Training
Start training with the following command:

```bash
bash run_training.sh
```

**Default Hyperparameters:**
- Base Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Epochs: 15
- Gradient Accumulation Steps: 8
- Latent Size: 8

---

## 📝 Citation

If you find this project or the ILVR framework useful, please cite our paper:

```bibtex
@article{dong2025ilvr,
  title={ILVR: Interleaved Latent Visual Reasoning with Selective Perceptual Modeling},
  author={Dong, Shuai and Wang, Siyuan and Liu, Xingyu and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2512.05665},
  year={2025}
}
```

Since we use the **CoMT dataset**, please also kindly cite the CoMT paper:

```bibtex
@article{comt2024benchmark,
  title={CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models},
  author={Wang, Siyuan and others},
  journal={arXiv preprint arXiv:2412.12932},
  year={2024}
}
```

## 🙏 Acknowledgements
This codebase is built upon [Qwen-VL](https://github.com/QwenLM/Qwen-VL) 、 [Transformers](https://github.com/huggingface/transformers) and [Mirage](https://github.com/UMass-Embodied-AGI/Mirage). We thank the authors for their open-source contributions.

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
```

# rlhf-gsm8k

GRPO (Group Relative Policy Optimization) experiments on GSM8K math reasoning, exploring reward design, hyperparameter sweeps, and data mixing strategies.

## 📦 Install

```bash
pip install git+https://github.com/tripathysagar/rlhf-gsm8k.git
```

## 📁 Project Structure

```
gsm8k_utils/                           # Pip-installable shared library
  ├── utils.py                         # Prompt formatting, answer extraction, evaluation
  └── grpo.py                          # GRPOExperiment class for reproducible runs
nbs/
  ├── 00_smollm2-135M-grpo-gsm8k.ipynb # Early experiment (SmolLM2-135M — too small)
  ├── 00_qwen2_5_0_5B_grpo_gsm8k.ipynb # Binary vs tiered reward ablation
  ├── 01_sft.ipynb                      # SFT warm-start (1024 GSM8K examples)
  └── 02_grpo_sweep.ipynb              # Hyperparameter sweep + multi-seed validation
```

## 🔑 Key Results

### Training Pipeline: Base → SFT → GRPO

| Stage | Accuracy | Δ |
|---|---|---|
| Base (Qwen2.5-0.5B) | 24.39% | — |
| + SFT (1 epoch, 1024 examples) | 31.36% | +6.97% |
| + GRPO (200 steps, best config) | **37.78% ± 2.61%** | **+6.42%** |

### Hyperparameter Sweep

Best config found (200 steps, binary reward 1.0/0.0):

| Param | Values Tested | Best |
|---|---|---|
| `num_generations` | 8, 16 | **16** |
| `beta` | 0.01, 0.02, 0.04 | **0.02** |
| `learning_rate` | 1e-5, 5e-5, 1e-4 | **5e-5** |
| `sft_frac` | 0.0, 0.5, 1.0 | **1.0** |

### Multi-Seed Validation

| Seed | Accuracy |
|---|---|
| 42 | 40.00% |
| 1337 | 37.27% |
| 7 | 36.06% |
| **Mean ± std** | **37.78% ± 2.61%** |

## 🤗 Models

| Model | Link |
|---|---|
| Base | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| SFT checkpoint | [tripathysagar/Qwen2.5-0.5B-GSM8K-SFT](https://huggingface.co/tripathysagar/Qwen2.5-0.5B-GSM8K-SFT) |

## 📝 Blog Post

Coming soon — covering the full journey from SFT to GRPO, reward design decisions, and lessons learned.

# rlhf-gsm8k

GRPO (Group Relative Policy Optimization) experiments on GSM8K math reasoning, exploring reward design, hyperparameter sweeps, and data mixing strategies.

## Project Structure

```
gsm8k_utils/          # Shared utilities (pip-installable)
  utils.py            # Prompt formatting, answer extraction, evaluation
nbs/
  00_smollm2-135M-grpo-gsm8k.ipynb   # Early GRPO experiment with SmolLM2-135M
  00_qwen2_5_0_5B_grpo_gsm8k.ipynb   # GRPO with Qwen2.5-0.5B + reward ablation
  01_sft.ipynb                         # SFT warm-start on GSM8K (1024 examples)
  02_grpo.ipynb                        # Parameterized GRPO sweep framework
```

## Install

```bash
pip install git+https://github.com/tripathysagar/rlhf-gsm8k.git
```

## Key Findings

### Binary vs Tiered Reward
| Reward | Pre-GRPO | Post-GRPO | Δ |
|---|---|---|---|
| Tiered (8.0/3.2/closeness) | 25.15% | 33.18% | +8.03% |
| Binary (1.0/0.0) | 25.76% | **35.91%** | **+10.15%** |

Simpler binary reward outperformed shaped reward — partial credit enabled mild reward hacking.

### Hyperparameter Sweep (100 steps, binary reward)
| Param | Values Tested | Best | Accuracy |
|---|---|---|---|
| `num_generations` | 8, 16 | **16** | 36.4%* |
| `beta` | 0.01, 0.02, 0.04 | **0.02** | 31.4% |
| `learning_rate` | 1e-5, 5e-5, 1e-4 | **5e-5** | 30.3% |
| `sft_frac` | 0.0, 0.5, 1.0 | **1.0** | 33.8% |

*\* High run-to-run variance — multiple seeds needed to confirm.*

## Models

- Base: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- SFT: [tripathysagar/Qwen2.5-0.5B-GSM8K-SFT](https://huggingface.co/tripathysagar/Qwen2.5-0.5B-GSM8K-SFT)
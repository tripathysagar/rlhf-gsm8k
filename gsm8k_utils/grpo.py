"""GRPO training pipeline for GSM8K experiments."""

import torch
import wandb
from types import SimpleNamespace
from datasets import load_dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, set_seed,
)
from peft import LoraConfig, get_peft_model
from gsm8k_utils.utils import extract_gold, format_prompt, extract_answer, perf_check

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CFG = dict(
    # Model
    model_id        = "tripathysagar/Qwen2.5-0.5B-GSM8K-SFT",
    lora_r          = 16,
    lora_alpha      = 32,

    # GRPO
    num_generations         = 8,
    max_completion_length   = 256,
    max_steps               = 100,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate           = 5e-5,
    beta                    = 0.04,
    warmup_steps            = 20,

    # wandb
    wandb_project   = "grpo-gsm8k",
    wandb_run_name  = "grpo-qwen2.5-0.5B-lora",

    # sft data mix
    grpo_train_size = 1024,
    sft_frac        = 0.5,

    # seed
    seed            = 1337,
)

# ── Device detection ──────────────────────────────────────────────────────────

def get_device_config():
    """Auto-detect GPU capabilities for dtype, attention, and precision."""
    if not torch.cuda.is_available():
        return dict(dtype=torch.float32, attn_impl="eager", bf16=False, fp16=False)
    cap = torch.cuda.get_device_capability()
    is_ampere_plus = cap[0] >= 8

    attn_impl = "sdpa"
    if is_ampere_plus:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError:
            pass

    return dict(
        dtype=torch.bfloat16 if is_ampere_plus else torch.float16,
        attn_impl=attn_impl,
        bf16=is_ampere_plus,
        fp16=not is_ampere_plus,
    )


# ── Reward ────────────────────────────────────────────────────────────────────

def reward_fn(completions, **kwargs):
    """Binary reward: 1.0 for correct answer, 0.0 otherwise."""
    golds = kwargs["gold"]
    rewards = []
    for comp, gold in zip(completions, golds):
        try:
            ans, _ = extract_answer(comp)
            gold_int = int(float(gold.replace(",", "")))
            rewards.append(1.0 if ans == gold_int else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# ── Experiment ────────────────────────────────────────────────────────────────

class GRPOExperiment:
    def __init__(self, cfg=None):
        cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg = SimpleNamespace(**cfg)

    def load_ds(self):
        test_ds = load_dataset("openai/gsm8k", "main", split="test")
        self.test_ds = test_ds.train_test_split(test_size=0.5, seed=42)["test"]

        all_train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
        if self.cfg.sft_frac == -1:
            self.train_ds = all_train
            return

        n_sft = int(self.cfg.grpo_train_size * self.cfg.sft_frac)
        n_new = self.cfg.grpo_train_size - n_sft

        sft_rows = all_train.select(range(self.cfg.grpo_train_size))
        new_rows = all_train.select(range(self.cfg.grpo_train_size, len(all_train)))

        sft_rows = sft_rows.shuffle(seed=42).select(range(n_sft)) if n_sft > 0 else sft_rows.select([])
        new_rows = new_rows.shuffle(seed=42).select(range(n_new)) if n_new > 0 else new_rows.select([])

        self.train_ds = concatenate_datasets([sft_rows, new_rows])

    def _fmt(self, ds):
        """Extract gold answers and format prompts."""
        ds = ds.map(extract_gold, remove_columns=["answer"]).rename_column("question", "prompt")
        return ds.map(lambda x: {"prompt": format_prompt(x["prompt"], self.tokenizer)})

    def fmt_ds(self):
        self.train_ds = self._fmt(self.train_ds)
        self.test_ds = self._fmt(self.test_ds)

    def setup_model(self):
        set_seed(self.cfg.seed)
        self.dev_cfg = get_device_config()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, dtype=self.dev_cfg["dtype"], device_map="auto",
            attn_implementation=self.dev_cfg["attn_impl"],
        )
        model.gradient_checkpointing_enable()
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

    def setup_trainer(self):
        wandb.init(
            project=self.cfg.wandb_project,
            name=f"{self.cfg.wandb_run_name}-seed{self.cfg.seed}",
            group=self.cfg.wandb_run_name,
            tags=["grpo", "multi-seed"],
            config=vars(self.cfg),
        )
        grpo_config = GRPOConfig(
            output_dir="grpo_qwen_gsm8k",
            num_generations=self.cfg.num_generations,
            max_completion_length=self.cfg.max_completion_length,
            max_steps=self.cfg.max_steps,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            logging_steps=1,
            report_to='wandb',
            beta=self.cfg.beta,
            lr_scheduler_type="cosine",
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=0.01,
            save_strategy="no",
            bf16=self.dev_cfg["bf16"],
            fp16=self.dev_cfg["fp16"],
        )
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_fn,
            args=grpo_config,
            train_dataset=self.train_ds,
            processing_class=self.tokenizer,
        )

    def train(self): self.trainer.train()

    def eval(self):
        self.model = self.model.merge_and_unload()
        accuracy, _ = perf_check(self.model, self.tokenizer, self.test_ds)
        wandb.log({"eval/accuracy": accuracy})
        return accuracy

    def cleanup(self):
        wandb.finish()
        del self.model, self.trainer
        torch.cuda.empty_cache()

    def __call__(self):
        self.load_ds()
        self.setup_model()
        self.fmt_ds()
        self.setup_trainer()
        self.train()
        result = self.eval()
        self.cleanup()
        return result
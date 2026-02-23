"""Shared utilities for GSM8K RLHF experiments."""

import torch
import regex as re
from tqdm import tqdm

# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then give your final answer as a single number on the last line in exact format"
    """\nThe answer is: {number}."""
)

# ── SFT helpers ─────────────────────────────────────────────────────────────

def clean_gold_answer(answer_text):
    """Strip <<...>> annotations and reformat with 'The answer is: N.' ending."""
    parts = answer_text.split("####")
    reasoning = parts[0].strip()
    final_num = parts[1].strip().replace(",", "") if len(parts) > 1 else ""
    reasoning = re.sub(r'<<.*?>>', '', reasoning)
    return f"{reasoning}\nThe answer is: {final_num}."


def make_sft_example(ex):
    """Convert a GSM8K example to SFT chat format."""
    cleaned = clean_gold_answer(ex['answer'])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": cleaned},
    ]
    return {"messages": messages}

# ── Prompt formatting ───────────────────────────────────────────────────────

def format_prompt(question, tokenizer):
    """Apply chat template to a question string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ── Answer extraction ───────────────────────────────────────────────────────

def extract_answer(response):
    """Extract numerical answer from model response.
    Returns (answer_int, used_format_bool) or (None, False)."""
    try:
        matches = re.findall(r'The answer is[:\s]*(\-?\d[\d,]*\.?\d*)', response, re.IGNORECASE)
        if matches:
            return int(float(matches[-1].replace(",", ""))), True
        nums = re.findall(r'\-?\d[\d,]*\.?\d*', response)
        if nums:
            return int(float(nums[-1].replace(",", ""))), False
    except Exception as e:
        print(f"[extract_answer error] response={response[:80]!r} err={e}")
    return None, False

# ── Evaluation ──────────────────────────────────────────────────────────────

def perf_check(model, tokenizer, test_data, batch_size=64):
    """Evaluate model accuracy on test_data (list of dicts with 'prompt' and 'gold' keys).
    Returns table_rows for display."""
    model.eval()
    tokenizer.padding_side = "left"

    correct, total = 0, 0
    results = []
    table_rows = []

    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i+batch_size]
        prompts = [ex["prompt"] for ex in batch]
        golds = [int(float(ex["gold"].replace(",", ""))) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)

        for j, (ids, gold_int) in enumerate(zip(out, golds)):
            response = tokenizer.decode(ids[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            ans, has_fmt = extract_answer(response)

            is_correct = (ans == gold_int)
            correct += int(is_correct)
            total += 1

            results.append({
                "gold": gold_int,
                "predicted": ans,
                "correct": is_correct,
                "response": response[:200],
            })
            table_rows.append(f"| {total} | {gold_int} | {ans} | {'✅ ' if is_correct else '❌ '} | {response[:80].replace(chr(10), ' ')} |")

    print(f"\nFinal Accuracy: {correct}/{total} = {correct/total:.2%}")

    model.train()
    tokenizer.padding_side = "right"

    return table_rows

# ── Inference helper ────────────────────────────────────────────────────────

def infer(model, tokenizer, question):
    """Generate a response for a single question."""
    model.eval()
    prompt = format_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256,
            do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    model.train()
    return response
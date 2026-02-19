import argparse
import json
import os
import re
import subprocess
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from datasets import load_dataset

# BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 16
REVISION = "b40621c1a4dcb65c065dffb0ee1c1298ce56d22e"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Given a task description and optional test cases, "
    "write a clean, correct Python function that solves the task. "
    "Only output the code, no explanations."
)

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8


def load_model(adapter_path=None):
    """Load base model with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-padding for batched generation

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map={"": 1}
    )
    model.generation_config = GenerationConfig.from_pretrained(BASE_MODEL)
    eos = model.generation_config.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    model.generation_config.pad_token_id = eos

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from: {adapter_path}")
    else:
        print("Using base model (no adapter)")

    model.eval()
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")
    return model, tokenizer


# def _build_prompt(tokenizer, task_description, test_list):
#     """Build a chat prompt string for a single task."""
#     user_content = task_description
#     if test_list:
#         user_content += "\n\nTest cases:\n" + "\n".join(test_list)

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user_content},
#     ]
#     return tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )

def _build_prompt(tokenizer, task_description, test_list):
    """Build a prompt string for a single task."""
    user_content = task_description
    if test_list:
        user_content += "\n\nTest cases:\n" + "\n".join(test_list)

    if tokenizer.chat_template:
        # Instruct model: use chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Base model: simple prompt format
        return f"### Instruction:\n{SYSTEM_PROMPT}\n\n### Input:\n{user_content}\n\n### Response:\n"

def _strip_code_fences(code):
    """Strip markdown code fences if present."""
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
    code = re.sub(r"\n?```\s*$", "", code.strip())
    return code


def generate_code(model, tokenizer, task_description, test_list, max_new_tokens=512):
    """Generate code for a single task description."""
    prompt = _build_prompt(tokenizer, task_description, test_list)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            top_p=None,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _strip_code_fences(generated_code)


def _generate_batch_inner(model, tokenizer, prompts, max_new_tokens=512):
    """Run model.generate on a list of prompt strings. Returns list of generated code."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Slice off the input portion for each sequence in the batch
    input_len = inputs["input_ids"].shape[1]
    generated_codes = []
    for i in range(output_ids.shape[0]):
        gen_ids = output_ids[i][input_len:]
        code = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_codes.append(_strip_code_fences(code))

    return generated_codes


def generate_code_batch(model, tokenizer, task_descriptions, test_lists, max_new_tokens=512):
    """Generate code for a batch of tasks at once, with OOM retry.

    On CUDA OOM, the batch is split in half and retried recursively.
    If a single sample still OOMs, it returns an empty string for that sample.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        task_descriptions: List of task description strings.
        test_lists: List of test_list (each a list of strings).
        max_new_tokens: Maximum tokens to generate per sample.

    Returns:
        List of generated code strings (one per task).
    """
    prompts = [
        _build_prompt(tokenizer, desc, tests)
        for desc, tests in zip(task_descriptions, test_lists)
    ]

    try:
        return _generate_batch_inner(model, tokenizer, prompts, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        n = len(prompts)
        if n == 1:
            print(f"  [WARNING] OOM on single sample, returning empty code")
            return [""]
        mid = n // 2
        print(f"  [WARNING] OOM with batch_size={n}, retrying as {mid} + {n - mid}")
        left = generate_code_batch(
            model, tokenizer, task_descriptions[:mid], test_lists[:mid], max_new_tokens
        )
        right = generate_code_batch(
            model, tokenizer, task_descriptions[mid:], test_lists[mid:], max_new_tokens
        )
        return left + right


def run_tests(generated_code, test_list, test_setup_code="", timeout=10):
    """Execute generated code against test cases. Returns (passed, total, details)."""
    results = []
    for test in test_list:
        script = ""
        if test_setup_code and test_setup_code.strip():
            script += test_setup_code.strip() + "\n\n"
        script += generated_code.strip() + "\n\n"
        script += test.strip() + "\n"

        try:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = result.returncode == 0
            error = result.stderr.strip() if not passed else ""
        except subprocess.TimeoutExpired:
            passed = False
            error = "Timeout"
        except Exception as e:
            passed = False
            error = str(e)

        results.append({"test": test, "passed": passed, "error": error})

    num_passed = sum(r["passed"] for r in results)
    return num_passed, len(results), results


def generate_and_test(model, tokenizer, example):
    """Generate code for a single MBPP example and run its unit tests."""
    task_desc = example["text"]
    test_list = example["test_list"]
    test_setup = example.get("test_setup_code", "")
    ground_truth = example["code"]

    print(f"\n{'='*60}")
    print(f"Task ID: {example['task_id']}")
    print(f"Description: {task_desc}")
    print(f"{'='*60}")

    generated_code = generate_code(model, tokenizer, task_desc, test_list)
    print(f"\nGenerated Code:\n{generated_code}")
    print(f"\nGround Truth:\n{ground_truth}")

    passed, total, details = run_tests(generated_code, test_list, test_setup)
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} passed")
    print(f"{'='*60}")
    for d in details:
        status = "PASS" if d["passed"] else "FAIL"
        print(f"  [{status}] {d['test']}")
        if d["error"]:
            print(f"         Error: {d['error'][:200]}")

    return passed, total, generated_code


def evaluate_on_split(model, tokenizer, split="test", max_samples=None, batch_size=BATCH_SIZE, save_results=None):
    """Evaluate pass@1 rate across a dataset split using batched generation.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        split: Which split to evaluate on ("test", "validation", "train").
        max_samples: Limit number of samples (None = all).
        batch_size: Number of samples to generate in parallel.
        save_results: Path to save per-task results as JSON (None = don't save).
    """
    ds = load_dataset("nlile/mbpp")
    data = ds[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    total_tasks = len(data)
    tasks_passed = 0
    all_results = []

    # Process in batches
    for batch_start in tqdm(range(0, total_tasks, batch_size), desc=f"Evaluating {split} (batch_size={batch_size})"):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_indices = list(range(batch_start, batch_end))
        batch_examples = [data[i] for i in batch_indices]

        # Batched generation
        task_descs = [ex["text"] for ex in batch_examples]
        test_lists = [ex["test_list"] for ex in batch_examples]
        generated_codes = generate_code_batch(model, tokenizer, task_descs, test_lists)

        # Run tests per example (still sequential — subprocess-based)
        for ex, generated_code in zip(batch_examples, generated_codes):
            passed, total, details = run_tests(
                generated_code,
                ex["test_list"],
                ex.get("test_setup_code", ""),
            )
            task_passed = passed == total
            if task_passed:
                tasks_passed += 1

            all_results.append({
                "task_id": ex["task_id"],
                "text": ex["text"],
                "generated_code": generated_code,
                "ground_truth": ex["code"],
                "tests_passed": passed,
                "tests_total": total,
                "all_passed": task_passed,
                "details": details,
            })

    pass_rate = tasks_passed / total_tasks * 100
    print(f"\n{'='*60}")
    print(f"Pass@1 on {split}: {tasks_passed}/{total_tasks} ({pass_rate:.1f}%)")
    print(f"{'='*60}")

    if save_results:
        with open(save_results, "w") as f:
            json.dump(
                {"pass_at_1": pass_rate, "passed": tasks_passed, "total": total_tasks, "results": all_results},
                f, indent=2,
            )
        print(f"Results saved to {save_results}")

    return tasks_passed, total_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on MBPP")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (local or HF hub). Omit for base model.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size for generation (default: {BATCH_SIZE})")
    parser.add_argument("--sample_index", type=int, default=None, help="If set, only test this single example index")
    parser.add_argument("--save", type=str, default=None, help="Save per-task results to this JSON file")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    if args.sample_index is not None:
        ds = load_dataset("nlile/mbpp")
        example = ds[args.split][args.sample_index]
        generate_and_test(model, tokenizer, example)
    else:
        evaluate_on_split(model, tokenizer, split=args.split, max_samples=args.max_samples, batch_size=args.batch_size, save_results=args.save)
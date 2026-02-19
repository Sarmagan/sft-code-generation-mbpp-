import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed, GenerationConfig
from peft import LoraConfig, PeftModel, get_peft_model
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "mbpp"
HF_USER = "SArmagan"

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Overall hyperparameters
EPOCHS = 1
BATCH_SIZE = 4
MAX_SEQUENCE_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 16

# QLoRA hyperparameters
LORA_R = 32 
LORA_ALPHA = LORA_R * 2
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.05

# Training hyperparameters
LEARNING_RATE = 3e-5 # 1e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.01
OPTIMIZER = "paged_adamw_32bit"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking
LOG_STEPS = 10
SAVE_STEPS = 20
LOG_TO_WANDB = True

# System prompt for code generation
# SYSTEM_PROMPT = (
#     "You are an expert Python programmer. Given a task description and optional test cases, "
#     "write a clean, correct Python function that solves the task. "
#     "Only output the code, no explanations."
# )

SYSTEM_PROMPT = (
    "You are an expert programmer. Given a programming task or question, "
    "provide a clear, correct function with explanation if needed."
)


def get_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    eos = model.generation_config.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    model.generation_config.pad_token_id = eos

    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")
    return model, tokenizer


def credentials():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token, add_to_git_credential=True)
    else:
        from huggingface_hub import login
        login()

    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"


# def format_evol_example(example, tokenizer):
#     """Format a single Evol-Instruct-Code example into chat-style prompt + completion."""
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": example["instruction"]},
#     ]
#     prompt = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )

#     return {"prompt": prompt, "completion": example["output"]}

def format_evol_example(example, tokenizer):
    prompt = f"### System:\n{SYSTEM_PROMPT}\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return {"prompt": prompt, "completion": example["output"]}

def prepare_datasets(seed=42, eval_ratio=0.05, max_samples=20000):
    """Load Evol-Instruct-Code-80k dataset and format for SFT training."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    print(f"Dataset size: {len(ds)}")

    # Filter out empty outputs
    ds = ds.filter(lambda x: x["output"] and x["output"].strip())
    print(f"After filtering empty outputs: {len(ds)}")

    # Shuffle and take max_samples
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        print(f"Using {max_samples} samples")

    def process_example(example):
        return format_evol_example(example, tokenizer)

    sft_data = []
    skipped = 0
    for idx, example in enumerate(ds):
        formatted = process_example(example)

        # Filter out examples that exceed MAX_SEQUENCE_LENGTH
        total_len = len(tokenizer.encode(formatted["prompt"] + " " + formatted["completion"]))
        if total_len > MAX_SEQUENCE_LENGTH:
            skipped += 1
            continue

        sft_data.append(formatted)
        if len(sft_data) == 1:
            print(f"\n{'='*60}")
            print(f"Sample example:")
            print(f"{'='*60}")
            print(f"PROMPT:\n{formatted['prompt'][:500]}")
            print(f"\nCOMPLETION:\n{formatted['completion'][:500]}")
            print(f"{'='*60}\n")

    print(f"Kept {len(sft_data)} examples, skipped {skipped} (exceeded {MAX_SEQUENCE_LENGTH} tokens)")

    # Write to JSONL then split
    out_path = "evol_sft.jsonl"
    with open(out_path, "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")

    dataset = load_dataset("json", data_files=out_path, split="train")
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)

    print(f"Train examples: {len(split['train'])}, Eval examples: {len(split['test'])}")

    return split["train"], split["test"]

def max_sequence_length_cutoff():
    """Analyze token lengths to choose MAX_SEQUENCE_LENGTH."""
    train_dataset, eval_dataset = prepare_datasets()
    _, tokenizer = get_model_and_tokenizer(BASE_MODEL)

    lengths = [
        len(tokenizer.encode(item["prompt"] + " " + item["completion"]))
        for item in train_dataset
    ]
    print(
        f"Mean: {np.mean(lengths):.0f}, "
        f"Max: {np.max(lengths)}, "
        f"95th percentile: {np.percentile(lengths, 95):.0f}, "
        f"99th percentile: {np.percentile(lengths, 99):.0f}"
    )


def SFT_with_QLoRA():
    """Fine-tune Llama on MBPP using SFT with QLoRA."""
    # credentials()
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL)

    train_dataset, eval_dataset = prepare_datasets()

    lora_parameters = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    train_parameters = SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb" if LOG_TO_WANDB else None,
        run_name=RUN_NAME,
        # max_length=MAX_SEQUENCE_LENGTH,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    fine_tuning = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_parameters,
        args=train_parameters,
    )

    torch.cuda.empty_cache()
    fine_tuning.train()
    fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")
    wandb.finish()


if __name__ == "__main__":
    # Preview dataset
    # prepare_mbpp_datasets() 

    # Check token lengths
    # max_sequence_length_cutoff()

    # Run training
    SFT_with_QLoRA()
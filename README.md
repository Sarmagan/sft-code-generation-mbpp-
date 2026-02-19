# Fine-Tuning Llama 3.2 3B for Code Generation with QLoRA

A parameter-efficient fine-tuning pipeline that adapts **Meta's Llama 3.2 3B** for code generation using **QLoRA** (Quantized Low-Rank Adaptation). The model is trained on the [Evol-Instruct-Code-80k](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) dataset and evaluated on the [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (Mostly Basic Python Problems) benchmark. Evol-Instruct-Code provides diverse, complex programming instructions that improve general code generation ability. MBPP serves as an independent, held-out benchmark

## Highlights

- **4-bit QLoRA** fine-tuning — trains a 3B-parameter model on consumer-grade GPUs
- **LoRA rank 32** applied to all attention + MLP projection layers for broad adaptation
- **Cosine learning rate schedule** with warmup for stable convergence
- **Weights & Biases** integration for experiment tracking
- Automatic **Hugging Face Hub** model upload after training

### Training Data

**Evol-Instruct-Code-80k-v1** — 80k instruction-output pairs covering a wide range of programming tasks, evolved from seed instructions using the Evol-Instruct methodology. The pipeline filters empty outputs and sequences exceeding the 512-token limit, then splits the remaining data into train/eval sets (95/5).

### Evaluation

The trained adapter is evaluated on **MBPP** (Mostly Basic Python Problems), a benchmark of ~1,000 crowd-sourced Python programming problems designed to test the ability of language models to synthesize short programs from natural language descriptions.

## Results

| Model | MBPP Pass@1 (test, 500 tasks) |
|---|---|
| Llama 3.2 3B Instruct (baseline) | **242 / 500 (48.4%)** |
| Llama 3.2 3B + QLoRA (this work) | 181 / 500 (36.2%) |

**Training config for this run:** 1 epoch, 7,272 training examples (randomly sampled from filtered Evol-Instruct-Code-80k).

### Analysis

The fine-tuned base model underperforms the Instruct variant by ~12 percentage points. This is expected given several factors:

- **Base vs. Instruct starting point.** The Instruct model benefits from Meta's full-scale instruction tuning across millions of examples. My QLoRA run trains the raw base model on only ~7k examples for a single epoch — a fraction of the data and compute used in the official post-training.
- **Limited training data.** 7,272 examples is a small subset; scaling to the full 80k dataset with multiple epochs would likely close the gap.
- **512-token sequence limit.** Many complex code solutions were filtered out due to small GPU limit, biasing training toward simpler examples.


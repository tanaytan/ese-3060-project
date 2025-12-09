# ESE 3060 Final Project Fall 2025

## Project Overview

This project contains two machine learning training benchmarks:

- **airbench94.py**: CIFAR-10 image classification benchmark
- **train_gpt.py**: GPT-2 training on the FineWeb-10B dataset

## Setup and Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (A100/H100 recommended)
- CUDA 11.7 or later

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Part 1

### Overview

CIFAR-10 training benchmark achieving 94.01% average accuracy in 3.83 seconds on an NVIDIA A100-SXM. NOTE: We used an NVIDIA A100-PCIe for our experiments.

The main file is:

- airbench94.py — CIFAR-10 training benchmark used to reproduce and modify the “airbench94” baseline described in the project handout.

My experiments (baseline and modifications) are logged in **cifar10/logs/** and summarized in **master_summary.csv**.

### Execution

The script supports command-line overrides for experiments.
Default behavior (no arguments):

- Runs 25 seeds
- Uses my modified TTA 1.5 settings
- Uses 9.9 training epochs

Basic run

```bash
python airbench94.py
```

Run with custom experiment name

```bash
python airbench94.py --experiment-name my_experiment
```

Change number of epochs

```bash
python airbench94.py --train-epochs 9.9
```

Change TTA level
(TTA 0 = no test-time augmentation, TTA 1 = flip only, TTA 2 = flip + 1 translation i.e. TTA 1.5)

```bash
python airbench94.py --tta-level 2
```

Change number of repeated runs (seeds)

```bash
python airbench94.py --num-runs 50
```

You can combine these:

```bash
python airbench94.py \
    --experiment-name br_crop_ep9.9 \
    --tta-level 2 \
    --train-epochs 9.9 \
    --num-runs 50
```

### Output

- Per-epoch training metrics (loss, accuracy)
- Validation and test-time augmentation (TTA) accuracy
- Mean and Std. statistics
- Logs saved to `logs/{uuid}/log.pt`
- Console log saved to `cifar10/logs/{experiment_name}_console.txt`

### Hardware Requirements

- NVIDIA A100-PCIe GPU to reproduce
- CUDA 11.7+
- NVIDIA Driver 515.105.01 or compatible

### Reference

Based on: [cifar10-airbench legacy airbench94.py](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench94.py)

## Part2

### Overview

This project evaluates architectural modifications to NanoGPT, specifically **Grouped Query
Attention (GQA)** and **Multi-Depthwise Head Attention (MDHA)**, to measure their impact on
training speed and validation loss. Experiments were run on the **FineWeb-10B** dataset using an
8×H100(SXM) setup and follow the standard 5100-step speedrun protocol.

GQA is opptional and disabled by default. The final configuration of GQA was selected through 250-step ablations tests:
- `num_kv_groups = 1`
- `num_gqa_layers = 8` (top transformer layers)

MDHA is optional and disabled by default due to its high compute cost. The final configuration used `mdha_kernel_size = 3`, which was based on the provided PRIMER paper.

---

### Data Preparation

Download and preprocess FineWeb-10B:

```bash
python cached_fineweb10B.py 9
```

This produces the sharded `.bin` and `.idx` files expected by `train_gpt.py`. You may have to move this into a `data` folder.

---

### Running Experiments

The different experiments can be run from the terminal using the following terminal commands with boolean flags and variation flags.

To run the baseline mode:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To run the baseline+GQA model:

```bash
USE_GQA=1 \
NUM_KV_GROUPS=1 \
NUM_GQA_LAYERS=8 \
USE_MDHA=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This is how the GQA 250-step ablations tests were conducted, varying the number of shared KV groups (`num_kv_groups`) from 1 to 3 and varying the number of layers (`num_gqa_layers`) shared from the top 4 to 12.

To run the baseline+GQA+MDHA model:

```bash
USE_GQA=1 \
USE_MDHA=1 \
NUM_KV_GROUPS=1 \
NUM_GQA_LAYERS=8 \
MDHA_KERNEL_SIZE=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All configuration values are printed at runtime for reproducibility. After the code is written to the log file, the configuration values are written to the log file before the train output is written.

---

### Hardware Requirements

- **8× NVIDIA H100 SXM GPUs** (project tested and benchmarked on this setup)
- PyTorch **2.4.1+** with CUDA **12.1**
- Python **3.10+**
- Sufficient disk space for FineWeb-10B shards

The runpod usage for this is shown as a png in `part2/logs` and mentioned in the corresponding appendices of this report.

---

### Logging and Outputs

The logs are stored in `part2/logs`. In the appendices of my report, I denote the naming convention, what each log file represents, and the configuration of each file. This is **extrememly important** as my experiments used modularized code so I would not have to change it each time.

Each run produces a log file containing:

- Per-step training loss  
- Step-average throughput  
- Cumulative training time  
- Final validation loss  

These logs were used to produce the final results comparisons, including:
- Baseline (3 seeds)
- GQA (3 seeds)
- GQA + MDHA (1 seed)
- 250-step ablations for selecting the final GQA layout

My plots and tables were generated by scripts in `part2/visualizations/scripts`, and have been manually classified into different subfolders inside `visualizaitons` and contextualized in the appendices of my report.

---

### Reference

This repository and training script are adapted from the
[modded-nanogpt speedrun record \#5](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt),
with additional instrumentation and architecture hooks for GQA and MDHA experiments.

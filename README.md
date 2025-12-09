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

## Running train_gpt.py

### Overview

Trains a GPT-2 model on the FineWeb-10B dataset. You will want to use an 8xH100.

### Execution

Download the data with

```bash
python cached_fineweb10B.py 9
```

and then run the script with

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Hardware Requirements

- Tested on 8× NVIDIA H100 80GB GPUs
- PyTorch 2.4.1+ with CUDA 12.1

### Reference

Based on: [modded-nanogpt record number #5](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt)

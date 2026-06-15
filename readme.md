# Benchmarking multi-class intracranial hemorrhage segmentation using semi-supervision

## Setup

### 1. Install Requirements
- Create a conda environment from the requirements file
- Install all necessary packages for benchmarking

```bash
conda create --name new_env --file requirements.txt 
```
or

```bash
conda create --name new_env --file requirements2.txt
```

### 2. Set Environment Variables
Create the required data folders and set nnUNet environment variables:

```bash
export nnUNet_raw=/home/user/Benchmarking_MBHSeg/data/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/user/Benchmarking_MBHSeg/data/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/user/Benchmarking_MBHSeg/data/nnUNet/nnUNet_results
```

## Available Models

Run models from `/home/user/Benchmarking_MBHSeg/nnUNet/nnunetv2/training/nnUNetTrainer/`:

- **nnUNetTrainer** — Default UNet model
- **nnUNetTrainer_CPS** — Cross Pseudo Supervision
- **nnUNetTrainer_DCNET** — Decoupled Consistency
- **nnUNetTrainer_FPML** — Feature perturbation enhanced mutual learning for semi-supervised segmentation
- **nnUNetTrainer_MLRPL** — Mutual Learning with Reliable Pseudo Labels
- **nnUNetTrainer_UAMT** — Uncertainty-aware mean teacher
- **nnUNetTrainer_FPML_challengeX** — FPML with additional unlabeled samples from MBHSeg2024 challenge (192 labeled + random unlabeled samples) 


## Data Preparation

1. Place your dataset in `data/nnUNet/nnUNet_raw/` following nnUNet format
2. Preprocess the data:

```bash
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity
```

## Training

### Step 1: Preprocess
```bash
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity
```

### Step 2: Train Model

**Command template:**
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train <dataset_id> 3d_fullres <fold> -tr <trainer_name> \
  --percentage_labeled_data <percentage> --seed_name <seed> --seed_balance <seed2>
```

**Parameters:**
- `<dataset_id>` — Dataset ID (e.g., `200`)
- `<fold>` — Fold index for cross-validation (e.g., `0`)
- `<trainer_name>` — Trainer to use (e.g., `nnUNetTrainer_FPML`)
- `<percentage>` — Labeled data percentage (e.g., `0.1` for 10%)
- `<seed>` — Random seed (e.g., `0`)
- `<seed2>` — Seed for labeled data selection (e.g., `42`)

**Example:**
```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres 0 -tr nnUNetTrainer_FPML \
  --percentage_labeled_data 0.1 --seed_name 0 --seed_balance 42
```

## Results

Model outputs are saved to the `nnUNet_results/` directory:
- Logs and checkpoints in `Dataset200/`
- Best model weights stored per fold and trainer


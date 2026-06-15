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


Running training:

First, preprocessing the data. 

```bash
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity 
```

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train d 3d_fullres f -tr trainerName --percentage_labeled_data p --seed_name s1 --seed_balance s2 
```
where:

- `d` — dataset identifier (ID)
- `f` — fold index (default: `0`)
- `trainerName` — name of the trainer
- `p` — percentage of the training set to use as labeled data
- `s1` — random seed (default: `0`)
- `s2` — random seed used for selecting the labeled subset

Example run: 

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 200 3d_fullres 0 -tr nnUNetTrainer_FPML --percentage_labeled_data 0.1 --seed_name 0 --seed_balance 42 
```
# Multimodal Auto-Completion: A New Task for Predicting User Input in Visually-Grounded Conversations

This repository contains code and datasets for evaluating multimodal vision-language models on multimodal auto-completion task.

## Repository Structure

```
MAC/
├── chatas/              # Evaluation metrics and datasets handling utilities
├── ms-swift-chatas/     # Swift integration for model evaluation and training
├── TAC_models/          # Model implementations for text auto-completion models
├── router/              # Router Implementation for model selection between TAC and MAC model
├── userstudy/           # User study web application for TAC and MAC models
├── ckpts/               # Model checkpoints (to be created)
├── data/                # Datasets (to be created)
└── environment.yaml     # Conda environment configuration
├── gpt_4v-filtering.txt # Filtering script for GPT-4V
├── README.md            # This README file
```

## Setup Instructions

### 1. Clone the repository

```bash
https://github.com/devichand579/MAC.git
```

### 2. Environment Setup

```bash
conda env create -f environment.yaml
conda activate MAC
```

### 3. Checkpoints and data setup


Download checkpoints and filtered datasets from here. Create the following directories for checkpoints and data:

```
├── ckpts
│   ├── Imagechat_ckpts
│   │   ├── Paligemma
│   │   ├── Qwen2_VL
│   │   ├── MiniCPM_V
│   │   └── MiniCPM_no_img
│   ├── MMDD_ckpts
│       ├ Paligemma
│       ├ Qwen2_VL
│       ├ MiniCPM_V
│       └ MiniCPM_no_img
│   
└── data
|   ├── MMDD
|   │   ├── images
|   │   └── train.csv
|   │   └── dev.csv
|   │   └── test.csv
|   └── ImageChat
        ├── image_chat
        └── yfcc_images

```
## Running inference for Multimodal Auto-Completion 

Navigate to the ms-swift-chatas directory and use the `run_eval.sh` script to run evaluations:

```bash
cd ms-swift-chatas
```

### Basic Usage

```bash
bash run_eval.sh <model_name> [batch_size] [outer_batch_size] [dataset] [split]
```

#### Available Models

- `paligemma`: PaliGemma 2 3B PT-224
- `qwen`: Qwen2-VL-2B-Instruct
- `minicpm_i`: MiniCPM-V-2_6
- `minicpm_noimg`: MiniCPM-V-2_6 (no images)

#### Available Datasets

- `image_chat`
- `mmdd`
#### Split range
- `0` to `7`
- Split was introduced to overcome computational limitations, to run on a complete dataset for any model, you must run the evaluation script 8 times with incremental split values and you will acquire outputs of completions in 8 different files named accordingly.
### Examples

1. Run PaliGemma with default settings:
```bash
bash run_eval.sh paligemma
```

2. Run Qwen with custom batch sizes:
```bash
bash run_eval.sh qwen 8 16
```

3. Run MiniCPM-V with ImageChat dataset:
```bash
bash run_eval.sh minicpm_i 8 16 image_chat 0
```

4. Run MiniCPM-V with MMDD dataset:
```bash
bash run_eval.sh minicpm_i 8 16 mmdd 0
```

### Output

Evaluation results will be saved in the `/output/` directory at root with filenames based on the model, dataset and split number used.

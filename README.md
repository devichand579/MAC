# Multimodal Auto-Completion: A New Task for Predicting User Input in Visually-Grounded Conversations

This repository contains code and datasets for evaluating multimodal vision-language models on multimodal auto-completion task.

## Repository Structure

```
MAC/
├── chatas/              # Evaluation metrics and datasets handling utilities
├── ms-swift-chatas/     # Swift integration for model evaluation and training
├── TAC_models/          # Model implementations for text auto-completion models
├── imagechat_outputs/   # Outputs for imagechat dataset
├── mmdd_outputs/        # Outputs for mmdd dataset
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

## Finetuning multimodal models for Multimodal Auto-Completion

Navigate to the ms-swift-chatas directory and use the `template_training.sh` script to finetune the models:

```bash
cd ms-swift-chatas
```
Go through the `template_training.sh` script and fill in the required parameters for finetuning the models.

## Training text auto-completion models for Multimodal Auto-Completion

Navigate to the TAC_models directory and go through the respective README.md files of MPC and QB for training the text auto-completion models.


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

NOTE: If you find any module name mismatch errors while loading the checkpoints, please update the adapter config file in the checkpoints directory or have a compatible version of the swift framework.

### Output

Evaluation results will be saved in the `/output/` directory at root with filenames based on the model, dataset and split number used.


## Computations of evaluation metrics for Multimodal Auto-Completion

### Creating intersection of dialogs for the datasets for comparison of results on seen and unseen data
 
 ```bash
 python chatas/code/utils/dataset.py
 ```

 After running this command, you will have the intersection of dialogs for the datasets in the following files:
 ```
 data/ImageChat/intersection_imagechat.json
 data/MMDD/intersection_mmdd.json
 ```

## Computing metrics 

```bash
python chatas/eval.py --input <output_file_path> --dataset <dataset_name> --data_path <data_file_path> --intersection_path <intersection_file_path>
```  

Example for ImageChat dataset:
```bash
python chatas/eval.py --input imagechat_outputs/imagechat.minicpm_image.csv --dataset image_chat --data_path data/ImageChat/image_chat/test.csv --intersection_path data/ImageChat/intersection_imagechat.json
```

Example for MMDD dataset:
```bash
python chatas/eval.py --input mmdd_outputs/mmdd.minicpm_image.csv --dataset mmdd --data_path data/MMDD/test.csv --intersection_path data/MMDD/intersection_mmdd.json
```
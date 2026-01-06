# Router-Suggest: A Router for Auto-Completion of User Input in Visually-Grounded Conversations [EACL 2026]

This repository contains code and datasets for evaluating multimodal vision-language models on multimodal auto-completion task and evalute Router-Suggest for model selection between TAC and MAC models.

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
|-- QB_ckpts/            # QB model checkpoints
└── environment.yaml     # Conda environment configuration
├── gpt_4v-filtering.txt # Filtering script for GPT-4V
|-- analyze_neural_classifier_results.py # Script to analyze the results of the neural classifier
|-- analyze_rf_results.py # Script to analyze the results of the random forest classifier
|-- neural_hyperparameter_mmdd.json # Results of neural classifier for Multimodal Auto-Completion with hyperparameter tuning for mmdd dataset
|-- neural_hyperparameter_imagechat.json # Results of neural classifier for Multimodal Auto-Completion with hyperparameter tuning for imagechat dataset
|-- rf_hyperparameter_mmdd.json # Results of random forest classifier for Multimodal Auto-Completion with hyperparameter tuning for mmdd dataset
|-- rf_hyperparameter_imagechat.json # Results of random forest classifier for Multimodal Auto-Completion with hyperparameter tuning for imagechat dataset
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

## Training and inference for text auto-completion models for Multimodal Auto-Completion

Navigate to the TAC_models directory and go through the respective README.md files of MPC and QB for training and inference of text auto-completion models.


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

### Computing metrics 

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

## Router Framework for Multimodal Auto-Completion

### Neural Classifier for Multimodal Auto-Completion

1. Creating the master dataset for training the router framework, Configure the models neeeded in `router/models/main.py` so that labels are generated for the models accordingly.
```bash
python router/utils/dataset_utils.py --dataset <dataset_name>
```
This will create two files in the root directory:
```
combined_pred_by_idx_<dataset_name>.json
master_dataset_<dataset_name>.csv
```

2. Creating the master dataset of features for training the router framework
```bash
python router/utils/create_latent_train.py --master_dataset_path <master_dataset_path> --model_name <model_name>
```
This will create files in the root directory:
```
master_dataset_<dataset_name>_embeddings.npy
master_dataset_<dataset_name>_embedding_mapping.csv
```

3. Fethcing the clock times for inference of the models
```bash
cd ms-swift-chatas
python benchmark_inference.py --use_vllm
```
This will create a file in the root directory:
```
benchmark_results.json
```

4. Training the router framework
```bash
python router/training/neural_classifier.py --dataset <dataset_name>  --use_cost_loss --lambda <lambda_weight>
```
Use the last two flags to enable cost-weighted loss and set the lambda weight for the loss. This will create a file in the root directory:
```
mlp_classifier_<dataset_name>.pth
mlp_label_encoder_<dataset_name>.pkl
```
This will print all the scores for the router framework with and without cost-weighted loss.
with hyperparameter tuning:
```bash
python router/training/neural_classifier.py --dataset <dataset_name> --use_hyperparameter_tuning
```
This will print all the scores for the router framework and save the results in the root directory.

5. Analyzing the results
```bash
python analyze_neural_classifier_results.py --input_json <results_file_path> --output_json <output_file_path>
```
This will print the best configuration for the router framework.

### Random Forest Classifier for Multimodal Auto-Completion

1. Creating the master dataset for training the router framework, Configure the models neeeded in `router/models/main.py` so that labels are generated for the models accordingly.
```bash
python router/utils/dataset_utils.py --dataset <dataset_name>
```
This will create two files in the root directory:
```
combined_pred_by_idx_<dataset_name>.json
master_dataset_<dataset_name>.csv
```
2. Creating the master dataset of features for training the router framework
```bash
python router/utils/create_train.py --master_dataset_path <master_dataset_path> --dataset <dataset_name> --run_models --combine_outputs
```
This will create files in the root directory:
```
master_dataset_<model_name>.csv
master_dataset_combined_<dataset_name>.csv
```

3. Training the router framework
```bash
python router/training/random_forest_train.py --data_path <data_path> --dataset <dataset_name>
```
This will create a file in the root directory:
```
random_forest_model.pkl
label_encoders.pkl
```
with hyperparameter tuning:
```bash
python router/training/random_forest_train.py --data_path <data_path> --dataset <dataset_name> --use_hyperparameter_tuning
```
This will print all the scores for the router framework and save the results in the root directory.

4. Analyzing the results
```bash
python analyze_rf_results.py --input_json <results_file_path> --output_json <output_file_path>
```
This will print the best configuration for the router framework.

## User Study for Multimodal Auto-Completion

A Flask-based web application for conducting user studies for TAC and MAC models.

## Requirements

- Python 3.10 or higher
- Flask
- Gunicorn (for production deployment)
- Additional dependencies needed for running swift framework

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd userstudy
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Note: The requirements.txt file includes only the basic dependencies. You may need to install additional packages depending on which models you want to use:


3. Prepare model checkpoints:
   - Ensure the `ckpt` directory contains the necessary model files for Query Blazer (QB) and MPC models

## Running the Application

### Running Locally

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. The application will be available at:
   ```
   http://localhost:5000/
   ```

3. To start a new session, navigate to:
   ```
   http://localhost:5000/?new_example=true
   ```

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t userstudy .
   ```

2. Run the container:
   ```bash
   docker run -p 8080:8080 userstudy
   ```

3. The application will be available at:
   ```
   http://localhost:8080/
   ```

## Application Structure

- `app.py`: Main application file containing the Flask server and model implementations
- `conv.py`: Contains conversation pools and utilities
- `models/`: Directory containing model implementations and utilities
- `static/`: Frontend files (HTML, CSS, JavaScript)
- `imagechat_samples/`: Sample data for image-based conversations

## Usage

1. When the application starts, it randomly selects a conversation context from the available pool
2. The MiniCPM model is used by default for text completion
3. The application logs user interactions in `session_logs.csv` and context information in `context_log.csv`


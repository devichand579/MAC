#!/bin/bash
# ====================================================================================
# Template Training Script for CHATAS Models
# ====================================================================================
# This template provides a standardized structure for training multimodal models
# with the Swift framework for CHATAS (Chat-as) tasks.
#
# USAGE:
#   1. Copy this template and rename it for your specific training task
#   2. Fill in the required parameters (marked with <...>)
#   3. Uncomment and adjust optional parameters as needed
#   4. Run the script: bash your_training_script.sh
# ====================================================================================

# === CONFIGURATION SECTION ===
# Set your WANDB API key for experiment tracking
WANDB_API_KEY=<your_wandb_api_key_here>

# === MODEL SELECTION ===
# Choose one of the following model types:
# - minicpmv2_6: MiniCPM-V-2_6
# - paligemma: Google's PaLI-Gemma models
# - qwen2_vl: Qwen2 Vision-Language models
MODEL_TYPE="<model_type>"

# Model path or Hugging Face ID
# Examples:
# - "openbmb/MiniCPM-V-2_6"
# - "google/paligemma2-3b-pt-224"
# - "qwen/Qwen2-VL-2B-Instruct"
MODEL_PATH="<model_path_or_hf_id>"

# === DATASET CONFIGURATION ===
# Dataset type for CHATAS
# Options: imgchat, mmdd, etc.
DATASET_TYPE="<dataset_type>"

# Paths to dataset and image directories
DATASET_DIR="<path_to_dataset_directory>"
IMAGE_DIR="<path_to_image_directory>"

# Output directory for saving model checkpoints and logs
OUTPUT_DIR="./exp_output_${MODEL_TYPE}_${DATASET_TYPE}"

# === TRAINING PARAMETERS ===
BATCH_SIZE=8
MAX_LENGTH=4196
NUM_EPOCHS=5

# === OPTIONAL PARAMETERS ===
# Uncomment and adjust as needed
LEARNING_RATE="5e-5"
# GRADIENT_ACCUMULATION_STEPS=4
# FP16_FLAG="--fp16"
# NO_IMG_FLAG="--no_img"
# DEEPSPEED_CONFIG="--deepspeed ds_config.json"

# ====================================================================================
# === TRAINING COMMAND ===
# ====================================================================================

swift sft \
  --model_type ${MODEL_TYPE} \
  --model ${MODEL_PATH} \
  --dataset place_holder.jsonl \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --max_length ${MAX_LENGTH} \
  --chatas ${DATASET_TYPE} \
  --dataset_dir "${DATASET_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --report_to wandb \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs ${NUM_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  ${NO_IMG_FLAG:-} \
  ${FP16_FLAG:-} \
  ${DEEPSPEED_CONFIG:-} \
  ${GRADIENT_ACCUMULATION_STEPS:+--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}}

# ====================================================================================
# === EXAMPLES ===
# ====================================================================================
# Example 1: Training MiniCPM-V-2_6 on ImageChat dataset
# MODEL_TYPE="minicpmv2_6"
# MODEL_PATH="openbmb/MiniCPM-V-2_6"
# DATASET_TYPE="imgchat"
# DATASET_DIR="../data/ImageChat/image_chat/"
# IMAGE_DIR="../data/ImageChat/yfcc_images/"
# NO_IMG_FLAG="--no_img"  # Uncomment to train without images
#
# Example 2: Training PaLI-Gemma on MMDD dataset
# MODEL_TYPE="paligemma"
# MODEL_PATH="google/paligemma2-3b-pt-224"
# DATASET_TYPE="mmdd"
# DATASET_DIR="../data/MMDD/test.csv"
# IMAGE_DIR="../data/MMDD/images/"
#
# Example 3: Training Qwen2-VL on ImageChat dataset
# MODEL_TYPE="qwen2_vl"
# MODEL_PATH="qwen/Qwen2-VL-2B-Instruct"
# DATASET_TYPE="imgchat"
# DATASET_DIR="../data/ImageChat/image_chat/test.csv"
# IMAGE_DIR="../data/ImageChat/yfcc_images/"

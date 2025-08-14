#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_eval.sh - Simple wrapper to run infer_chatas.py with different models
# ------------------------------------------------------------------------------

set -e

# Show usage if no arguments or help flag
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 <model_name> [batch_size] [outer_batch_size] [dataset] [dataset_path] [image_dir]"
  echo ""
  echo "Models:"
  echo "  paligemma                - PaliGemma 2 3B PT-224"
  echo "  qwen                     - Qwen2-VL-2B-Instruct"
  echo "  minicpm_i                - MiniCPM-V-2_6"
  echo ""
  echo "  batch_size               - Inner batch size (default varies by model)"
  echo "  outer_batch_size         - Outer batch size (default varies by model)"
  echo "  dataset                  - Dataset to use (image_chat or mmdd, default: image_chat)"
  echo "  dataset_path             - Path to dataset file (default: ../../anubhab/ParlAI/data/image_chat/test.csv)"
  echo "  image_dir                - Path to image directory (default: ../../anubhab/ParlAI/data/yfcc_images)"
  echo ""
  echo "Examples:"
  echo "  bash $0 paligemma         # Run PaliGemma with default settings"
  echo "  bash $0 qwen 16 4         # Run Qwen with batch_size=16, outer_batch_size=4"
  echo "  bash $0 minicpm_i 8 16 image_chat /path/to/dataset.csv /path/to/images  # Run MiniCPM with imagechat dataset"
  exit 1
fi

MODEL_NAME="$1"
BATCH_SIZE="${2:-16}"  # Default to 16 if not provided
OUTER_BATCH="${3:-32}" # Default to 32 if not provided
DATASET="${4:-image_chat}" # Default to image_chat if not provided
DATASET_PATH="${5:-../../anubhab/ParlAI/data/image_chat/test.csv}" # Default path
IMAGE_DIR="${6:-../../anubhab/ParlAI/data/yfcc_images}" # Default image directory

case "$MODEL_NAME" in
  paligemma)
    OUTPUT_FILE="out.all.imagechat.paligemma2.3b.pt224"
    ADAPTER="ckpts/exp_output_paligemma_imgchat/v3-20250529-040720/checkpoint-105000"
    MODEL="google/paligemma2-3b-pt-224"
    ;;
  qwen)
    OUTPUT_FILE="out.all.imagechat.qwen2.vl.2b.instruct"
    ADAPTER="ckpts/exp_output_qwen2_vl_imagechat/v1-20250529-055538/checkpoint-103000"
    MODEL="qwen/Qwen2-VL-2B-Instruct"
    ;;
  minicpm_i)
    OUTPUT_FILE="out.all.imagechat.minicpm_image"
    ADAPTER="ckpts/MiniCPM-V-2_6_ck_92000"
    MODEL="openbmb/MiniCPM-V-2_6"
    ;;
  *)
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: paligemma, qwen, minicpm_i"
    exit 1
    ;;
esac

echo "Running $MODEL_NAME evaluation:"
echo "  Model:           $MODEL"
echo "  Adapter:         $ADAPTER"
echo "  Output file:     $OUTPUT_FILE"
echo "  Batch size:      $BATCH_SIZE"
echo "  Outer batch:     $OUTER_BATCH"
echo "  Dataset:         $DATASET"
echo "  Dataset path:    $DATASET_PATH"
echo "  Image directory: $IMAGE_DIR"
echo ""


# Initialize conda for this script
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate swift

python infer_chatas.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --output_file "$OUTPUT_FILE" \
  --batch_size "$BATCH_SIZE" \
  --outer_batch_size "$OUTER_BATCH" \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --image_dir "$IMAGE_DIR"

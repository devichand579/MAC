#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_eval.sh - Simple wrapper to run infer_chatas.py with different models
# ------------------------------------------------------------------------------

set -e

# Show usage if no arguments or help flag
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: $0 <model_name> [batch_size] [outer_batch_size] [dataset] [split] [--resume]"
  echo ""
  echo "Models:"
  echo "  paligemma                - PaliGemma 2 3B PT-224"
  echo "  qwen                     - Qwen2-VL-2B-Instruct"
  echo "  minicpm_i                - MiniCPM-V-2_6"
  echo " minicpm_noimg            - MiniCPM-V-2_6 (no images)"
  echo ""
  echo "  batch_size               - Inner batch size (default varies by model)"
  echo "  outer_batch_size         - Outer batch size (default varies by model)"
  echo "  dataset                  - Dataset to use (image_chat or mmdd, default: image_chat)"
  echo "  split                    - Split number (default: 0)"
  echo "  --resume                 - Resume from existing output file (optional flag)"
  echo ""
  echo "Examples:"
  echo "  bash $0 paligemma         # Run PaliGemma with default settings"
  echo "  bash $0 qwen 16 4         # Run Qwen with batch_size=16, outer_batch_size=4"
  echo "  bash $0 minicpm_i 8 16 image_chat 0 --resume  # Run MiniCPM with resume flag"
  exit 1
fi

MODEL_NAME="$1"
BATCH_SIZE="${2:-16}"  # Default to 16 if not provided
OUTER_BATCH="${3:-32}" # Default to 32 if not provided
DATASET="${4:-image_chat}" # Default to image_chat if not provided
SPLIT="${5:-0}" # Default to 0 if not provided

# Check if --resume flag is provided
RESUME_FLAG=""
for arg in "$@"; do
  if [[ "$arg" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    break
  fi
done

case "$MODEL_NAME" in
  paligemma)
    MODEL="google/paligemma2-3b-pt-224"
    ;;
  qwen)
    MODEL="qwen/Qwen2-VL-2B-Instruct"
    ;;
  minicpm_i)
    MODEL="openbmb/MiniCPM-V-2_6"
    ;;
  minicpm_noimg)
    MODEL="openbmb/MiniCPM-V-2_6"
    ;;
  *)
    echo "Error: Unknown model '$MODEL_NAME'"
    echo "Available models: paligemma, qwen, minicpm_i, minicpm_noimg"
    exit 1
    ;;
esac

case "$DATASET" in
  image_chat)
    DATASET_PATH="../data/ImageChat/image_chat/test.csv"
    IMAGE_DIR="../data/ImageChat/yfcc_images"
    case "$MODEL_NAME" in
      paligemma) 
        OUTPUT_FILE="out.all.imagechat.paligemma2.3b.pt224"
        ADAPTER="../ckpts/Imagechat_ckpts/Paligemma"
        ;;
      qwen) 
        OUTPUT_FILE="out.all.imagechat.qwen2.vl.2b.instruct"
        ADAPTER="../ckpts/Imagechat_ckpts/Qwen2_VL"
        ;;
      minicpm_i) 
        OUTPUT_FILE="out.all.imagechat.minicpm_image"
        ADAPTER="../ckpts/Imagechat_ckpts/MiniCPM_V"
        ;;
      minicpm_noimg) 
        OUTPUT_FILE="out.all.imagechat.minicpm_noimg"
        ADAPTER="../ckpts/Imagechat_ckpts/MiniCPM_no_img"
        ;;
    esac
    ;;
  mmdd) 
    DATASET_PATH="../data/MMDD/test.csv"
    IMAGE_DIR="../data/MMDD/images"
    case "$MODEL_NAME" in
      paligemma) 
        OUTPUT_FILE="out.all.mmdd.paligemma2.3b.pt224"
        ADAPTER="../ckpts/MMDD_ckpts/Paligemma"
        ;;
      qwen) 
        OUTPUT_FILE="out.all.mmdd.qwen2.vl.2b.instruct"
        ADAPTER="../ckpts/MMDD_ckpts/Qwen2_VL"
        ;;
      minicpm_i) 
        OUTPUT_FILE="out.all.mmdd.minicpm_image"
        ADAPTER="../ckpts/MMDD_ckpts/MiniCPM_V"
        ;;
      minicpm_noimg) 
        OUTPUT_FILE="out.all.mmdd.minicpm_noimg"
        ADAPTER="../ckpts/MMDD_ckpts/MiniCPM_no_img"
        ;;
    esac
    ;;
  *)
    echo "Error: Unknown dataset '$DATASET'"
    echo "Available datasets: image_chat, mmdd"
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
echo "  Split:           $SPLIT"
echo "  Resume:          $([ -n "$RESUME_FLAG" ] && echo "True" || echo "False")"
echo ""


# Initialize conda for this script
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate MAC

python infer_chatas.py \
  --resume\
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --output_file "$OUTPUT_FILE" \
  --batch_size "$BATCH_SIZE" \
  --outer_batch_size "$OUTER_BATCH" \
  --dataset "$DATASET" \
  --dataset_path "$DATASET_PATH" \
  --image_dir "$IMAGE_DIR" \
  --split "$SPLIT" \
  $RESUME_FLAG

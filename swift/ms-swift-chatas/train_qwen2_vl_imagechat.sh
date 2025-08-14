# == Tushar ==
WANDB_API_KEY=092d260784db89780a6b4d51d28d98584b0cd07b \
swift sft \
  --model_type qwen2_vl \
  --model qwen/Qwen2-VL-2B-Instruct \
  --dataset place_holder.jsonl \
  --per_device_train_batch_size 8 \
  --max_length 4196 \
  --chatas imgchat \
  --dataset_dir "anubhab/ParlAI/data/image_chat/" \
  --image_dir "anubhab/ParlAI/data/yfcc_images/" \
  --report_to wandb \
  --output_dir ./exp_output_qwen2_vl_imagechat \
  --num_train_epochs 5 \
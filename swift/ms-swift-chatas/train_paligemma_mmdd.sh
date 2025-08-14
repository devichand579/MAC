# == Internal ==
WANDB_API_KEY=092d260784db89780a6b4d51d28d98584b0cd07b swift sft \
  --model_type paligemma \
  --model google/paligemma2-3b-pt-224 \
  --dataset place_holder.jsonl \
  --per_device_train_batch_size 8 \
  --max_length 4196 \
  --chatas mmdd \
  --dataset_dir "../../CHAT-AS-MULTIMODAL/data/MMDD/" \
  --image_dir "../../CHAT-AS-MULTIMODAL/data/MMDD/images/" \
  --report_to wandb \
  --output_dir ./exp_output_paligemma_mmdd \


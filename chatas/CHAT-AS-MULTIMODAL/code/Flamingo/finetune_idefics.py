import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text as idefics, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig as bnb
import torchvision.transforms as transforms
from utils.dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "HuggingFaceM4/idefics-9b"

#load the model in 4-bit precision 
bnb_config = bnb(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)
processor = AutoProcessor.from_pretrained(model_name)

# inference to check if it works
model = idefics.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

print(model)

def model_inference(model, processor, prompts, max_new_tokens=64):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=max_new_tokens, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    
# Example to try from
image_path_by_url = create_image_path_by_url(
    image_names_dir="../../tmp/image_names", 
    images_dir="../../tmp/images"
)

dialog_data = DialogCCData(
    path="../data/DialogCC/test.csv",
    to_filter=True,
    to_replace=True,
    image_path_by_url=image_path_by_url,
    to_unroll=True,
    min_images_per_dialog=1,
    n_samples=1100,
    to_split=True
)
print(f"Total dialogs: {len(dialog_data)}")





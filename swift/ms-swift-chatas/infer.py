import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Set the number of GPUs to use

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)  # Import necessary modules

from swift.utils import seed_everything  # Set random seed
import torch

model_type = ModelType.minicpm_v_v2_5_chat
template_type = get_default_template_type(model_type)  # Obtain the template type, primarily used for constructing special tokens and image processing workflow
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_id_or_path='/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5',
                                       model_kwargs={'device_map': 'auto'})  # Load the model, set model type, model path, model parameters, device allocation, etc., computation precision, etc.
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)  # Construct the template based on the template type
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']  # Image URL
query = '距离各城市多远？'  # Note: Query is still in Chinese, consider translating if needed
response, history = inference(model, template, query, images=images)  # Obtain results through inference
print(f'query: {query}')
print(f'response: {response}')

# Streaming output
query = '距离最远的城市是哪？'  # Note: Query is still in Chinese, consider translating if needed
gen = inference_stream(model, template, query, history, images=images)  # Call the streaming output interface
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')

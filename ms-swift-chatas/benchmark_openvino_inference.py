# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Benchmark script to measure absolute clock times for OpenVINO model inference.
Tests Qwen2-VL-2B-Instruct OpenVINO model on a mix of MMDD and ImageChat samples.
"""

import os
import sys
import time
import warnings
from typing import List, Dict, Tuple
import statistics
from PIL import Image
import torch
import numpy as np

from tqdm import tqdm

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU RAM tracking will be disabled.")

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Suppress tokenizer parallelism warnings

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatas.code.utils.dataset import (
    Dialog,
    MMDDData,
    create_image_path_by_url_mmdd,
    ImageChatData,
    create_image_path_by_url_image_chat,
)
from argparse import ArgumentParser
import json


def transform_dialog_data_to_message(dialog: Dialog, suffix: str) -> dict:
    """Transform dialog to message format"""
    query = ""
    images = []
    for utterance in dialog.utterances[:-1]:
        if len(utterance.images) > 0:
            images.extend(utterance.images)
            query += f"{utterance.speaker}:<|IMAGE|>\n"
        else:
            query += f"{utterance.speaker}:{utterance.text}\n"
    images.extend(dialog.utterances[-1].images)
    query += f"{dialog.utterances[-1].speaker}:{dialog.utterances[-1].text}"
    
    return {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": images[0] if images else None,
            },
            {"type": "text", "text": query},
        ],
    }


def get_samples(
    mmdd_path: str, 
    mmdd_image_dir: str,
    imagechat_path: str,
    imagechat_image_dir: str,
    n_mmdd: int = 50,
    n_imagechat: int = 50
) -> List[Tuple[str, dict, str]]:
    """Get mixed samples from both datasets"""
    data = []
    
    # Load MMDD samples
    print(f"Loading {n_mmdd} samples from MMDD...")
    mmdd_data = MMDDData(
        path=mmdd_path,
        to_filter=True,
        to_replace=True,
        image_path_by_url=create_image_path_by_url_mmdd(mmdd_image_dir),
        to_unroll=True,
        min_images_per_dialog=1,
        to_split=True,
    )
    
    mmdd_samples = []
    for dialog, suffix in mmdd_data:
        if len([i for i in dialog.utterances[:-1] if len(i.images) > 0]) == 0:
            continue
        mmdd_samples.append((dialog.idx, transform_dialog_data_to_message(dialog, suffix), "mmdd"))
        if len(mmdd_samples) >= n_mmdd:
            break
    
    # Load ImageChat samples
    print(f"Loading {n_imagechat} samples from ImageChat...")
    imagechat_data = ImageChatData(
        path=imagechat_path,
        to_filter=True,
        to_replace=True,
        image_path_by_url=create_image_path_by_url_image_chat(imagechat_image_dir),
        to_unroll=True,
        min_images_per_dialog=1,
        to_split=True,
    )
    
    imagechat_samples = []
    for dialog, suffix in imagechat_data:
        if len([i for i in dialog.utterances[:-1] if len(i.images) > 0]) == 0:
            continue
        imagechat_samples.append((dialog.idx, transform_dialog_data_to_message(dialog, suffix), "imagechat"))
        if len(imagechat_samples) >= n_imagechat:
            break
    
    data = mmdd_samples + imagechat_samples
    print(f"Loaded {len(data)} total samples ({len(mmdd_samples)} MMDD, {len(imagechat_samples)} ImageChat)")
    return data


def get_cpu_memory_gb() -> float:
    """Get current CPU RAM usage in GB"""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB


def prepare_openvino_inputs(message: dict, processor):
    """Prepare inputs for OpenVINO model from message format"""
    # Extract image and text from message
    image_path = None
    text = None
    
    for content_item in message.get("content", []):
        if content_item.get("type") == "image":
            image_path = content_item.get("image")
        elif content_item.get("type") == "text":
            text = content_item.get("text")
    
    # Load and process image if available
    image = None
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    
    # For Qwen2-VL, use the processor's expected format
    # Ensure text is a valid string
    if not text:
        text = ""
    
    if image is not None:
        # Replace <|IMAGE|> with Qwen2-VL's image token format
        # Qwen2-VL processor expects <image> token in text
        if "<|IMAGE|>" in text:
            text = text.replace("<|IMAGE|>", "<image>")
        # Ensure <image> token is present - add at beginning if not found
        if "<image>" not in text:
            text = "<image>\n" + text
        
        # Use processor with text containing <image> token and images list
        # Qwen2-VL processor should handle <image> token conversion
        # Try both list and string formats
        try:
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        except (TypeError, ValueError):
            # Fallback to string format
            inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
        
        # Verify and fix image tokens if needed
        # Qwen2-VL uses image_token_id = 151655
        image_token_id = 151655
        
        # First, get the exact number of image tokens needed from the image processor
        # This must match the number of image embeddings that will be produced
        num_image_tokens = None
        if image is not None:
            try:
                image_inputs = processor.image_processor(
                    images=[image], return_tensors='pt', do_resize=False
                )
                if 'image_grid_thw' in image_inputs:
                    image_grid_thw = image_inputs['image_grid_thw']
                    merge_size = getattr(processor.image_processor, 'merge_size', 14)
                    
                    # Handle different formats of image_grid_thw
                    if isinstance(image_grid_thw, torch.Tensor):
                        if image_grid_thw.dim() > 1:
                            grid_vals = image_grid_thw[0].cpu().numpy()
                        else:
                            grid_vals = image_grid_thw.cpu().numpy()
                    else:
                        grid_vals = image_grid_thw[0] if isinstance(image_grid_thw, (list, tuple)) and len(image_grid_thw) > 0 else image_grid_thw
                    
                    # Calculate: prod(grid_vals) / (merge_size^2)
                    if isinstance(grid_vals, np.ndarray):
                        prod = np.prod(grid_vals)
                    elif isinstance(grid_vals, (list, tuple)):
                        prod = np.prod(grid_vals)
                    else:
                        prod = float(grid_vals)
                    
                    num_image_tokens = int(prod // (merge_size ** 2))
                    num_image_tokens = max(1, num_image_tokens)
            except Exception as e:
                # If calculation fails, we'll use a default
                pass
        
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            # Check if image tokens are present
            if isinstance(input_ids, torch.Tensor):
                has_image_tokens = (input_ids == image_token_id).any().item()
                input_ids_np = input_ids.cpu().numpy()
            else:
                input_ids_np = np.array(input_ids)
                has_image_tokens = (input_ids_np == image_token_id).any()
            
            if image is not None:
                # Get the correct number of image tokens needed
                token_len = num_image_tokens if num_image_tokens is not None else 414
                
                # Convert to list for manipulation
                if isinstance(input_ids, torch.Tensor):
                    input_ids_list = input_ids_np.flatten().tolist()
                    was_tensor = True
                    dtype = input_ids.dtype
                else:
                    input_ids_list = input_ids_np.flatten().tolist()
                    was_tensor = False
                
                if not has_image_tokens:
                    # Image tokens missing - insert them at the beginning
                    # Insert after first token (usually BOS)
                    insert_pos = 1 if len(input_ids_list) > 0 else 0
                    input_ids_list = (
                        input_ids_list[:insert_pos] + 
                        [image_token_id] * token_len + 
                        input_ids_list[insert_pos:]
                    )
                else:
                    # Image tokens exist but count might be wrong - fix the count
                    # Count existing image tokens
                    existing_count = sum(1 for tok in input_ids_list if tok == image_token_id)
                    if existing_count != token_len:
                        # Remove existing image tokens and insert correct number
                        input_ids_list = [tok for tok in input_ids_list if tok != image_token_id]
                        # Insert correct number at the beginning
                        insert_pos = 1 if len(input_ids_list) > 0 else 0
                        input_ids_list = (
                            input_ids_list[:insert_pos] + 
                            [image_token_id] * token_len + 
                            input_ids_list[insert_pos:]
                        )
                
                # Convert back to tensor/array
                # Note: shape may have changed due to token insertion
                new_input_ids_array = np.array(input_ids_list)
                
                # Reshape to [1, seq_len] for batch size 1
                if len(new_input_ids_array.shape) == 1:
                    new_input_ids_array = new_input_ids_array.reshape(1, -1)
                
                # Convert back to tensor
                if isinstance(input_ids, torch.Tensor):
                    inputs['input_ids'] = torch.from_numpy(new_input_ids_array).to(input_ids.dtype)
                else:
                    inputs['input_ids'] = new_input_ids_array.tolist()
                
                # Also update attention_mask if present (it needs to match new length)
                if 'attention_mask' in inputs:
                    new_seq_len = new_input_ids_array.shape[1]
                    if isinstance(inputs['attention_mask'], torch.Tensor):
                        # Extend attention mask to match new length
                        old_mask = inputs['attention_mask']
                        if old_mask.shape[1] < new_seq_len:
                            # Pad with 1s (attending to new tokens)
                            pad_len = new_seq_len - old_mask.shape[1]
                            padding = torch.ones(old_mask.shape[0], pad_len, dtype=old_mask.dtype)
                            inputs['attention_mask'] = torch.cat([old_mask, padding], dim=1)
                        elif old_mask.shape[1] > new_seq_len:
                            # Truncate (shouldn't happen, but handle it)
                            inputs['attention_mask'] = old_mask[:, :new_seq_len]
                    else:
                        # Handle list format
                        old_mask = inputs['attention_mask']
                        if len(old_mask[0]) < new_seq_len:
                            pad_len = new_seq_len - len(old_mask[0])
                            for i in range(len(old_mask)):
                                old_mask[i].extend([1] * pad_len)
    else:
        # Text only - remove image placeholders if present
        text = text.replace("<|IMAGE|>", "").replace("<image>", "").strip()
        if not text:
            text = " "  # Empty string might cause issues, use space
        inputs = processor(text=text, return_tensors="pt", padding=True)
    
    # Ensure all tensors are on CPU (OpenVINO runs on CPU)
    # Handle both single tensors and lists/tuples
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            processed_inputs[k] = v.cpu()
        elif isinstance(v, (list, tuple)):
            processed_inputs[k] = [
                item.cpu() if isinstance(item, torch.Tensor) else item 
                for item in v
            ]
        else:
            processed_inputs[k] = v
    
    return processed_inputs


def benchmark_openvino_model(
    model_name: str,
    model_id: str,
    samples: List[Tuple[str, dict, str]],
    warmup_samples: int = 5,
    eval_samples: int = 10,
) -> Dict:
    """Benchmark OpenVINO model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Model: {model_id}")
    print(f"Engine: OpenVINO")
    print(f"{'='*60}")
    
    # Track CPU memory usage
    initial_memory_gb = get_cpu_memory_gb()
    
    # Initialize model
    print("Loading model...")
    load_start = time.time()
    
    try:
        from optimum.intel import OVModelForVisualCausalLM
        from transformers import AutoProcessor
        
        # Load model and processor
        model = OVModelForVisualCausalLM.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Track memory after model load
        post_load_memory_gb = get_cpu_memory_gb()
        model_memory_gb = post_load_memory_gb - initial_memory_gb
        
        # Fix device attribute if it's a string (OpenVINO models sometimes have device as string)
        # The generate method expects model.device.type to exist and match input tensor device
        if isinstance(model.device, str):
            # Create a device-like object with a type attribute that matches CPU (where inputs are)
            class DeviceWrapper:
                def __init__(self):
                    self.type = 'cpu'  # OpenVINO runs on CPU
                def __str__(self):
                    return 'cpu'
                def __repr__(self):
                    return "device(type='cpu')"
            model.device = DeviceWrapper()
        elif hasattr(model.device, 'type') and model.device.type != 'cpu':
            # Ensure device is CPU for OpenVINO
            model.device = torch.device('cpu')
        
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Track peak memory during warmup and inference
    peak_memory_gb = post_load_memory_gb
    
    # Warmup with samples
    if warmup_samples > 0 and len(samples) >= warmup_samples:
        print(f"Warming up with {warmup_samples} samples (one by one)...")
        try:
            for i in range(warmup_samples):
                sample = samples[i]
                message = sample[1]
                inputs = prepare_openvino_inputs(message, processor)
                _ = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                # Track peak memory during warmup
                current_memory = get_cpu_memory_gb()
                peak_memory_gb = max(peak_memory_gb, current_memory)
        except Exception as e:
            print(f"Warning: Warmup failed: {e}")
            import traceback
            traceback.print_exc()
        print("Warmup complete")
    
    # Evaluation with samples
    eval_count = min(eval_samples, len(samples))
    print(f"Running inference on {eval_count} samples (one by one, no batching)...")
    
    # Skip warmup samples and take eval_samples for evaluation
    eval_samples_list = samples[warmup_samples:warmup_samples + eval_count]
    
    # Measure total inference time
    inference_times = []
    total_start = time.time()
    
    for sample in tqdm(eval_samples_list, desc=f"Inferencing {model_name}"):
        sample_start = time.time()
        message = sample[1]
        
        try:
            # Prepare inputs
            inputs = prepare_openvino_inputs(message, processor)
            
            # Run inference
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            
            # Track peak memory during inference
            current_memory = get_cpu_memory_gb()
            peak_memory_gb = max(peak_memory_gb, current_memory)
            
            sample_time = time.time() - sample_start
            inference_times.append(sample_time)
        except Exception as e:
            print(f"Error during inference: {e}")
            # Use a large time for failed samples
            inference_times.append(float('inf'))
    
    total_time = time.time() - total_start
    
    # Filter out failed samples (inf times)
    valid_times = [t for t in inference_times if t != float('inf')]
    if len(valid_times) == 0:
        print("Error: All inference samples failed!")
        return None
    
    # Calculate statistics
    total_samples = len(valid_times)
    avg_time_per_sample = total_time / total_samples
    avg_time_per_sample_individual = statistics.mean(valid_times)
    min_sample_time = min(valid_times)
    max_sample_time = max(valid_times)
    std_sample_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
    
    throughput = total_samples / total_time  # samples per second
    
    # Calculate minimum memory required (peak memory during inference)
    min_memory_required_gb = peak_memory_gb
    
    results = {
        "model_name": model_name,
        "model_id": model_id,
        "total_samples": total_samples,
        "warmup_samples": warmup_samples,
        "load_time_seconds": load_time,
        "total_inference_time_seconds": total_time,
        "average_time_per_sample_seconds": avg_time_per_sample,
        "average_time_per_sample_individual_seconds": avg_time_per_sample_individual,
        "min_sample_time_seconds": min_sample_time,
        "max_sample_time_seconds": max_sample_time,
        "std_sample_time_seconds": std_sample_time,
        "throughput_samples_per_second": throughput,
        "sample_times": valid_times,
        "cpu_memory_initial_gb": initial_memory_gb,
        "cpu_memory_post_load_gb": post_load_memory_gb,
        "cpu_memory_model_gb": model_memory_gb,
        "cpu_memory_peak_gb": peak_memory_gb,
        "cpu_memory_min_required_gb": min_memory_required_gb,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results for {model_name}:")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Warmup samples:          {warmup_samples}")
    print(f"Load time:               {load_time:.2f} seconds")
    print(f"Total inference time:    {total_time:.2f} seconds")
    print(f"Average per sample:      {avg_time_per_sample:.4f} seconds")
    print(f"Average (individual):    {avg_time_per_sample_individual:.4f} seconds")
    print(f"Min sample time:         {min_sample_time:.4f} seconds")
    print(f"Max sample time:         {max_sample_time:.4f} seconds")
    print(f"Std dev sample time:     {std_sample_time:.4f} seconds")
    print(f"Throughput:              {throughput:.2f} samples/second")
    print(f"\nCPU Memory Usage:")
    print(f"Initial memory:          {initial_memory_gb:.2f} GB")
    print(f"Model memory:             {model_memory_gb:.2f} GB")
    print(f"Peak memory:              {peak_memory_gb:.2f} GB")
    print(f"Min memory required:     {min_memory_required_gb:.2f} GB")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = ArgumentParser(description="Benchmark inference times for OpenVINO model")
    parser.add_argument("--mmdd_path", type=str, default="../data/MMDD/test.csv",
                       help="Path to MMDD dataset")
    parser.add_argument("--mmdd_image_dir", type=str, default="../data/MMDD/images",
                       help="MMDD image directory")
    parser.add_argument("--imagechat_path", type=str, default="../data/ImageChat/image_chat/test.csv",
                       help="Path to ImageChat dataset")
    parser.add_argument("--imagechat_image_dir", type=str, default="../data/ImageChat/yfcc_images",
                       help="ImageChat image directory")
    parser.add_argument("--n_mmdd", type=int, default=50,
                       help="Number of MMDD samples")
    parser.add_argument("--n_imagechat", type=int, default=50,
                       help="Number of ImageChat samples")
    parser.add_argument("--warmup_samples", type=int, default=5,
                       help="Number of samples for warmup (processed one by one)")
    parser.add_argument("--eval_samples", type=int, default=10,
                       help="Number of samples for evaluation (processed one by one)")
    parser.add_argument("--model_id", type=str, default="Qwen2-VL-2B-Instruct-openvino",
                       help="OpenVINO model ID from HuggingFace")
    parser.add_argument("--output_file", type=str, default="benchmark_openvino_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Load samples
    print("="*60)
    print("Loading benchmark samples...")
    print("="*60)
    samples = get_samples(
        args.mmdd_path,
        args.mmdd_image_dir,
        args.imagechat_path,
        args.imagechat_image_dir,
        n_mmdd=args.n_mmdd,
        n_imagechat=args.n_imagechat
    )
    
    # Benchmark OpenVINO model
    results = benchmark_openvino_model(
        model_name="qwen2-vl-2b-openvino",
        model_id=args.model_id,
        samples=samples,
        warmup_samples=args.warmup_samples,
        eval_samples=args.eval_samples
    )
    
    if results:
        # Save results
        output_path = args.output_file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nBenchmarking failed. No results saved.")


if __name__ == "__main__":
    main()


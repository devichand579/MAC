import os
import sys
import time
import warnings
from typing import List, Dict, Tuple
import statistics

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Suppress tokenizer parallelism warnings

# Apply flash_attn patch as early as possible, before any imports that might use it
# This ensures the patch is applied in the main process before vLLM workers are spawned
try:
    import flash_attn
    import torch
    
    # Patch flash_attn_varlen_func to ensure cu_seqlens_q is on CUDA
    original_func = flash_attn.flash_attn_interface.flash_attn_varlen_func
    
    def patched_flash_attn_varlen_func(*args, **kwargs):
        args_list = list(args)
        # Fix cu_seqlens_q - typically 4th argument
        if len(args_list) >= 4:
            cu_seqlens_q = args_list[3]
            if cu_seqlens_q is not None and isinstance(cu_seqlens_q, torch.Tensor):
                if not cu_seqlens_q.is_cuda and len(args_list) > 0:
                    q = args_list[0]
                    if isinstance(q, torch.Tensor) and q.is_cuda:
                        args_list[3] = cu_seqlens_q.to(q.device)
        # Fix in kwargs too
        if 'cu_seqlens_q' in kwargs:
            cu_seqlens_q = kwargs['cu_seqlens_q']
            if cu_seqlens_q is not None and isinstance(cu_seqlens_q, torch.Tensor):
                if not cu_seqlens_q.is_cuda and len(args_list) > 0:
                    q = args_list[0]
                    if isinstance(q, torch.Tensor) and q.is_cuda:
                        kwargs['cu_seqlens_q'] = cu_seqlens_q.to(q.device)
        return original_func(*args_list, **kwargs)
    
    flash_attn.flash_attn_interface.flash_attn_varlen_func = patched_flash_attn_varlen_func
except Exception:
    pass  # If flash_attn is not available, continue

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatas.code.utils.dataset import (
    Dialog,
    MMDDData,
    create_image_path_by_url_mmdd,
    ImageChatData,
    create_image_path_by_url_image_chat,
)
from swift.llm import (
    InferEngine,
    InferRequest,
    PtEngine,
    VllmEngine,
    RequestConfig,
)
from argparse import ArgumentParser
import torch
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


def benchmark_model(
    model_name: str,
    model_path: str,
    adapter_path: str,
    samples: List[Tuple[str, dict, str]],
    warmup_samples: int = 5,
    eval_samples: int = 10,
    use_quantization: bool = False,
    use_vllm: bool = False
) -> Dict:
    """Benchmark a single model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Model: {model_path}")
    print(f"Adapter: {adapter_path}")
    print(f"Engine: {'vLLM' if use_vllm else 'PyTorch'}")
    print(f"{'='*60}")
    
    # Initialize model
    print("Loading model...")
    load_start = time.time()
    
    if model_path == "google/paligemma2-3b-pt-224":
        torch._dynamo.config.disable = True
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Setup quantization if enabled (vLLM has its own quantization)
    quantization_config = None
    if use_quantization and not use_vllm:  # vLLM quantization is different
        try:
            from swift.llm import QuantizationConfig
            quantization_config = QuantizationConfig(
                quantization_method='bnb',
                quantization_bit=4
            )
            print("Using 4-bit quantization for faster inference")
        except Exception as e:
            print(f"Warning: Could not enable quantization: {e}")
    
    # Try vLLM first if requested
    if use_vllm:
        try:
            # Patch flash attention BEFORE importing vLLM to fix device placement issues
            import os
            os.environ["VLLM_USE_FLASH_ATTN"] = "0"
            os.environ["DISABLE_FLASH_ATTN"] = "1"
            # Try to patch flash_attn to ensure cu_seqlens_q is on CUDA
            # Note: This is already patched in vllm_engine.py, but we do it here too for extra safety
            try:
                import flash_attn
                # Check if already patched
                if not hasattr(flash_attn.flash_attn_interface.flash_attn_varlen_func, '_patched'):
                    original_flash_attn_varlen_func = flash_attn.flash_attn_interface.flash_attn_varlen_func
                    def patched_flash_attn_varlen_func(*args, **kwargs):
                        # Ensure cu_seqlens_q and cu_seqlens_k are on CUDA
                        args_list = list(args)
                        # cu_seqlens_q is typically the 4th positional argument (q, k, v, cu_seqlens_q, ...)
                        if len(args_list) >= 4:
                            cu_seqlens_q = args_list[3]
                            if cu_seqlens_q is not None and hasattr(cu_seqlens_q, 'is_cuda') and not cu_seqlens_q.is_cuda:
                                args_list[3] = cu_seqlens_q.cuda()
                        if len(args_list) >= 5:
                            cu_seqlens_k = args_list[4]
                            if cu_seqlens_k is not None and hasattr(cu_seqlens_k, 'is_cuda') and not cu_seqlens_k.is_cuda:
                                args_list[4] = cu_seqlens_k.cuda()
                        # Also check kwargs
                        if 'cu_seqlens_q' in kwargs:
                            cu_seqlens_q = kwargs['cu_seqlens_q']
                            if cu_seqlens_q is not None and hasattr(cu_seqlens_q, 'is_cuda') and not cu_seqlens_q.is_cuda:
                                kwargs['cu_seqlens_q'] = cu_seqlens_q.cuda()
                        if 'cu_seqlens_k' in kwargs:
                            cu_seqlens_k = kwargs['cu_seqlens_k']
                            if cu_seqlens_k is not None and hasattr(cu_seqlens_k, 'is_cuda') and not cu_seqlens_k.is_cuda:
                                kwargs['cu_seqlens_k'] = cu_seqlens_k.cuda()
                        return original_flash_attn_varlen_func(*args_list, **kwargs)
                    patched_flash_attn_varlen_func._patched = True
                    flash_attn.flash_attn_interface.flash_attn_varlen_func = patched_flash_attn_varlen_func
                    print("Patched flash_attn_varlen_func to ensure CUDA device placement")
            except Exception as patch_error:
                print(f"Warning: Could not patch flash_attn: {patch_error}")
            
            # Patch VllmEngine._prepare_engine_kwargs to handle device parameter issue
            from swift.llm.infer.infer_engine import vllm_engine
            import inspect
            from vllm import AsyncEngineArgs, EngineArgs
            
            # Store original method
            original_prepare = vllm_engine.VllmEngine._prepare_engine_kwargs
            
            def patched_prepare_engine_kwargs(self,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                max_model_len=None,
                max_num_seqs=256,
                disable_custom_all_reduce=False,
                enforce_eager=False,
                limit_mm_per_prompt=None,
                device='auto',
                enable_lora=False,
                max_loras=1,
                max_lora_rank=16,
                enable_prefix_caching=False,
                distributed_executor_backend=None,
                enable_sleep_mode=False,
                engine_kwargs=None,
            ):
                """Patched version that excludes device parameter if not supported"""
                if engine_kwargs is None:
                    engine_kwargs = {}
                disable_log_stats = engine_kwargs.pop('disable_log_stats', True)
                
                if self.use_async_engine:
                    engine_cls = AsyncEngineArgs
                else:
                    engine_cls = EngineArgs
                
                # Check if parameters are supported
                parameters = inspect.signature(engine_cls).parameters
                
                # Only add disable_log_requests if supported
                if self.use_async_engine and 'disable_log_requests' in parameters:
                    engine_kwargs['disable_log_requests'] = True
                
                # Handle enable_lora, limit_mm_per_prompt, enable_sleep_mode like original
                if 'enable_lora' in parameters and enable_lora:
                    engine_kwargs['enable_lora'] = enable_lora
                    engine_kwargs['max_loras'] = max_loras
                    engine_kwargs['max_lora_rank'] = max_lora_rank
                else:
                    assert not enable_lora, 'The current version of vLLM does not support `enable_lora`. Please upgrade vLLM.'
                
                if 'limit_mm_per_prompt' in parameters and limit_mm_per_prompt:
                    engine_kwargs['limit_mm_per_prompt'] = limit_mm_per_prompt
                else:
                    assert not limit_mm_per_prompt, (
                        'The current version of VLLM does not support `limit_mm_per_prompt`. Please upgrade VLLM.')
                if 'enable_sleep_mode' in parameters:
                    engine_kwargs['enable_sleep_mode'] = enable_sleep_mode
                
                model_info = self.model_info
                if self.config.architectures is None:
                    architectures = {'deepseek_vl2': ['DeepseekVLV2ForCausalLM']}[self.model_meta.model_type]
                    engine_kwargs['hf_overrides'] = {'architectures': architectures}
                
                # Build engine_args dict, excluding device if not supported
                engine_args_kwargs = {
                    'model': self.model_dir,
                    'dtype': vllm_engine.dtype_mapping[model_info.torch_dtype],
                    'gpu_memory_utilization': gpu_memory_utilization,
                    'tensor_parallel_size': tensor_parallel_size,
                    'pipeline_parallel_size': pipeline_parallel_size,
                    'max_model_len': max_model_len,
                    'max_num_seqs': max_num_seqs,
                    'disable_log_stats': disable_log_stats,
                    'disable_custom_all_reduce': disable_custom_all_reduce,
                    'enforce_eager': enforce_eager,
                    'trust_remote_code': True,
                    'enable_prefix_caching': enable_prefix_caching,
                    'distributed_executor_backend': distributed_executor_backend,
                    **engine_kwargs,
                }
                
                # Only add device if it's supported
                if 'device' in parameters:
                    engine_args_kwargs['device'] = device
                
                engine_args = engine_cls(**engine_args_kwargs)
                
                # Set device attribute for swift internal use even if not in params
                if 'device' not in parameters:
                    engine_args.device = device
                
                if distributed_executor_backend == 'external_launcher':
                    engine_args.disable_custom_all_reduce = True
                self.engine_args = engine_args
                self.enable_lora = enable_lora
                if max_model_len is not None:
                    model_info.max_model_len = max_model_len
                
            # Apply patch temporarily
            vllm_engine.VllmEngine._prepare_engine_kwargs = patched_prepare_engine_kwargs
            
            try:
                # Clear GPU cache before vLLM initialization to ensure consistent memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Give a moment for memory to stabilize
                    time.sleep(0.5)
                
                # Check available GPUs
                num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
                print(f"Detected {num_gpus} GPU(s) available")
                
                # For multimodal models, tensor parallelism can cause device placement issues
                # Start with single GPU, can enable TP later if needed
                # Tensor parallelism with multimodal models can have cu_seqlens_q device issues
                tensor_parallel_size = 1  # Disable TP for now to avoid device placement issues
                if num_gpus >= 2:
                    print(f"Note: Tensor parallelism disabled for multimodal models to avoid device placement issues")
                
                # Calculate gpu_memory_utilization dynamically based on actual free memory
                # vLLM checks: free_memory >= (gpu_memory_utilization * total_memory)
                # So we need: gpu_memory_utilization <= free_memory / total_memory
                if torch.cuda.is_available():
                    # Get memory info for the first GPU
                    free_mem, total_mem = torch.cuda.mem_get_info(0)
                    free_mem_gb = free_mem / (1024**3)
                    total_mem_gb = total_mem / (1024**3)
                    
                    # Calculate maximum safe utilization: use 85% of free memory
                    # This leaves some headroom for other operations
                    max_safe_util = (free_mem_gb * 0.85) / total_mem_gb
                    
                    # Clamp between 0.2 and 0.9 to ensure reasonable values
                    gpu_mem_util = max(0.2, min(0.9, max_safe_util))
                    
                    print(f"GPU memory: {total_mem_gb:.2f} GB total, {free_mem_gb:.2f} GB free")
                    print(f"Using {gpu_mem_util:.1%} GPU memory utilization per GPU (tensor_parallel_size={tensor_parallel_size})")
                    print(f"  -> This will use ~{gpu_mem_util * total_mem_gb:.2f} GB, which fits in {free_mem_gb:.2f} GB free")
                else:
                    gpu_mem_util = 0.5
                    print(f"CUDA not available, using default {gpu_mem_util:.1%} GPU memory utilization")
                
                # Check if model supports LoRA in vLLM
                # PaliGemma doesn't support LoRA in vLLM, so disable it
                model_supports_lora = "paligemma" not in model_path.lower()
                use_lora_for_vllm = bool(adapter_path) and model_supports_lora
                
                if adapter_path and not model_supports_lora:
                    print(f"Warning: vLLM does not support LoRA for {model_path}. LoRA will be disabled for vLLM.")
                
                engine = VllmEngine(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    gpu_memory_utilization=gpu_mem_util,  # Explicitly set to avoid default 0.9
                    tensor_parallel_size=tensor_parallel_size,  # Split model across GPUs
                    max_num_seqs=1,  # One at a time
                    max_model_len=2048,  # Increased for multimodal models (images + text need more tokens)
                    enable_lora=use_lora_for_vllm,  # Disable for models that don't support it
                    max_loras=1 if use_lora_for_vllm else 0,
                    max_lora_rank=16,
                    enforce_eager=True,  # Use eager mode to avoid flash attention device placement issues
                    use_async_engine=False,  # Use sync engine to avoid async engine initialization issues
                )
                # Load adapter if provided (vLLM adapter loading)
                if adapter_path:
                    print(f"Note: vLLM adapter should be loaded automatically if compatible")
            finally:
                # Restore original method
                vllm_engine.VllmEngine._prepare_engine_kwargs = original_prepare
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Check for specific vLLM error types
            if error_type == "EngineGenerateError" or "EngineGenerateError" in error_msg:
                print(f"Warning: vLLM encountered an internal engine error during inference.")
                print(f"This is a known issue with vLLM's async engine. Falling back to PtEngine...")
            elif "_log_task_completion" in error_msg:
                print(f"Warning: vLLM version compatibility issue with swift library.")
                print(f"This is a known issue with vLLM version mismatch.")
                print(f"Falling back to PtEngine...")
            elif "Flash-Attention version" in error_msg or "flash-attn" in error_msg.lower():
                print(f"Warning: Could not use vLLM due to Flash-Attention version mismatch.")
                print(f"vLLM requires Flash-Attention >=2.7.1,<=2.8.2, but got 2.8.3.")
                print(f"To fix: pip install flash-attn==2.8.2")
                print(f"Falling back to PtEngine...")
            elif "memory profiling" in error_msg.lower() or "inconsistent GPU memory" in error_msg.lower() or "Initial free memory" in error_msg:
                print(f"Warning: Could not use vLLM due to GPU memory inconsistency during initialization.")
                print(f"This can happen when other processes are using GPU memory concurrently.")
                print(f"vLLM is sensitive to memory changes during profiling. Falling back to PtEngine...")
            else:
                print(f"Warning: Could not use vLLM, falling back to PtEngine: {e}")
                import traceback
                traceback.print_exc()
            use_vllm = False
    
    # Fallback to PtEngine (or use it if vLLM not requested)
    if not use_vllm:
        try:
            engine = PtEngine(
                model_path,
                max_batch_size=1,  # Process one sample at a time
                adapters=[adapter_path] if adapter_path else None,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_impl="flash_attn",
                quantization_config=quantization_config,
            )
        except Exception as e:
            print(f"Warning: Could not use flash_attn, falling back to default: {e}")
            engine = PtEngine(
                model_path,
                max_batch_size=1,  # Process one sample at a time
                adapters=[adapter_path] if adapter_path else None,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )

    
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Warmup with 5 samples, processing one by one
    request_config = RequestConfig(max_tokens=64, temperature=0)
    if warmup_samples > 0 and len(samples) >= warmup_samples:
        print(f"Warming up with {warmup_samples} samples (one by one)...")
        try:
            for i in range(warmup_samples):
                sample = samples[i]
                warmup_request = InferRequest(messages=[sample[1]])
                _ = engine.infer([warmup_request], request_config)
        except Exception as e:
            error_type = type(e).__name__
            if error_type == "EngineGenerateError" or "EngineGenerateError" in str(e) or "outputs_queue" in str(e):
                print(f"\nError: vLLM engine failed during warmup inference.")
                print(f"Trying to fix by reinitializing vLLM engine...")
                # Clean up vLLM engine
                try:
                    del engine
                except:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    time.sleep(1.0)  # Give time for cleanup
                
                # Try reinitializing vLLM with sync engine (use_async_engine=False)
                try:
                    print(f"Reinitializing vLLM with sync engine (use_async_engine=False)...")
                    engine = VllmEngine(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        gpu_memory_utilization=gpu_mem_util,
                        tensor_parallel_size=tensor_parallel_size,
                        max_num_seqs=1,
                        max_model_len=2048,
                        enable_lora=use_lora_for_vllm,
                        max_loras=1 if use_lora_for_vllm else 0,
                        max_lora_rank=16,
                        enforce_eager=False,
                        use_async_engine=False,  # Force sync engine
                    )
                    print(f"vLLM sync engine initialized. Retrying warmup...")
                    # Retry warmup with sync engine
                    for i in range(warmup_samples):
                        sample = samples[i]
                        warmup_request = InferRequest(messages=[sample[1]])
                        _ = engine.infer([warmup_request], request_config)
                    print(f"vLLM sync engine working successfully!")
                except Exception as e2:
                    print(f"vLLM sync engine also failed: {e2}")
                    print(f"Falling back to PtEngine...")
                    # Clean up again
                    try:
                        del engine
                    except:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                    
                    # Fallback to PtEngine
                    use_vllm = False
                    engine = PtEngine(
                        model_path,
                        max_batch_size=1,
                        adapters=[adapter_path] if adapter_path else None,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        quantization_config=quantization_config,
                    )
                    print(f"PtEngine initialized. Retrying warmup...")
                    # Retry warmup with PtEngine
                    for i in range(warmup_samples):
                        sample = samples[i]
                        warmup_request = InferRequest(messages=[sample[1]])
                        _ = engine.infer([warmup_request], request_config)
            else:
                raise
        print("Warmup complete")
    
    # Evaluation with 10 samples, processing one by one
    eval_count = min(eval_samples, len(samples))
    print(f"Running inference on {eval_count} samples (one by one, no batching)...")
    
    # Skip warmup samples and take eval_samples for evaluation
    eval_samples_list = samples[warmup_samples:warmup_samples + eval_count]
    
    # Measure total inference time
    inference_times = []
    total_start = time.time()
    
    for sample in tqdm(eval_samples_list, desc=f"Inferencing {model_name}"):
        sample_start = time.time()
        infer_request = InferRequest(messages=[sample[1]])
        
        # Run inference sample by sample
        resp_list = engine.infer([infer_request], request_config)
        
        sample_time = time.time() - sample_start
        inference_times.append(sample_time)
    
    # Single sync at the end for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    total_samples = eval_count
    avg_time_per_sample = total_time / total_samples
    avg_time_per_sample_individual = statistics.mean(inference_times)
    min_sample_time = min(inference_times)
    max_sample_time = max(inference_times)
    std_sample_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
    
    throughput = total_samples / total_time  # samples per second
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "adapter_path": adapter_path,
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
        "sample_times": inference_times,
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
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = ArgumentParser(description="Benchmark inference times for multiple models")
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
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization for faster inference (may reduce accuracy slightly)")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM engine instead of PyTorch (much faster, 2-5x speedup)")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Model configurations
    models_config = [
        {
            "name": "paligemma",
            "model_path": "google/paligemma2-3b-pt-224",
            "adapter_mmdd": "../ckpts/MMDD_ckpts/Paligemma",
            "adapter_imagechat": "../ckpts/Imagechat_ckpts/Paligemma",
        },
        {
            "name": "qwen",
            "model_path": "qwen/Qwen2-VL-2B-Instruct",
            "adapter_mmdd": "../ckpts/MMDD_ckpts/Qwen2_VL",
            "adapter_imagechat": "../ckpts/Imagechat_ckpts/Qwen2_VL",
        },
        {
            "name": "minicpm_i",
            "model_path": "openbmb/MiniCPM-V-2_6",
            "adapter_mmdd": "../ckpts/Imagechat_ckpts/MiniCPM_V",
            "adapter_imagechat": "../ckpts/Imagechat_ckpts/MiniCPM_V",
        },
    ]
    
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
    
    # Separate samples by dataset
    mmdd_samples = [(idx, msg, ds) for idx, msg, ds in samples if ds == "mmdd"]
    imagechat_samples = [(idx, msg, ds) for idx, msg, ds in samples if ds == "imagechat"]
    
    all_results = {}
    
    # Benchmark each model
    for model_config in models_config:
        model_name = model_config["name"]
        model_path = model_config["model_path"]
        
        # Use adapter based on dataset mix (use MMDD adapter as default, or average both)
        # For mixed dataset, we'll use the adapter that matches the majority or test both
        adapter_path = model_config["adapter_imagechat"]  # Default to MMDD
        
        # Benchmark on mixed samples
        results = benchmark_model(
            model_name=model_name,
            model_path=model_path,
            adapter_path=adapter_path,
            samples=samples,
            warmup_samples=args.warmup_samples,
            eval_samples=args.eval_samples,
            use_quantization=args.use_quantization,
            use_vllm=args.use_vllm
        )
        
        all_results[model_name] = results
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Total Time (s)':<18} {'Avg/Sample (s)':<18} {'Throughput (samples/s)':<25}")
    print("-"*60)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['total_inference_time_seconds']:<18.2f} "
              f"{results['average_time_per_sample_seconds']:<18.4f} "
              f"{results['throughput_samples_per_second']:<25.2f}")
    
    print("="*60)
    
    # Save results
    output_path = args.output_file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Find fastest model
    fastest = min(all_results.items(), key=lambda x: x[1]['total_inference_time_seconds'])
    print(f"\nFastest model: {fastest[0]} ({fastest[1]['total_inference_time_seconds']:.2f} seconds)")


if __name__ == "__main__":
    main()


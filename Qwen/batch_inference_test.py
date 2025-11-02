import torch
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
import random
from typing import List, Dict, Tuple
import shutil
from datetime import datetime
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

class ConversationDataset(Dataset):
    """Dataset for conversation samples"""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class Qwen2_5_VL_MultiGPU:
    def __init__(self, rank, world_size, model_path="Merged-Qwen2.5-VL-7B", 
                 max_new_tokens=1024):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://', 
                                  world_size=world_size, rank=rank)
        
        # Load model on specific GPU
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use fp16 for better performance
            device_map=self.device
        ).to(self.device)
        
        # Initialize processor
        max_pixels = 100352
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        
        self.gen_config = {
            "max_new_tokens": max_new_tokens,
        }
    
    def batch_chat(self, conversations: List[Dict], batch_size: int = 4) -> List[List[str]]:
        """Process multiple conversations in batches on a single GPU"""
        all_responses = []
        
        for i in range(0, len(conversations), batch_size):
            batch = conversations[i:i + batch_size]
            batch_responses = self._process_batch(batch)
            all_responses.extend(batch_responses)
            
            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_responses
    
    def _process_batch(self, batch: List[Dict]) -> List[List[str]]:
        """Process a batch of conversations"""
        batch_messages = []
        batch_histories = []
        batch_images = []
        
        # Prepare batch data
        for conv in batch:
            messages = conv["messages"]
            images = conv.get("images", [])
            
            # Convert conversation to multi-turn format
            history = []
            for j in range(0, len(messages), 2):
                user_msg = messages[j]
                if user_msg["role"] == "user":
                    # For first message, include images
                    if j == 0 and len(images) > 0:
                        content = []
                        for img_path in images:
                            content.append({"type": "image", "image": img_path})
                        content.append({"type": "text", "text": user_msg["content"].replace("<image>", "").strip()})
                        user_entry = {"role": "user", "content": content}
                    else:
                        user_entry = {"role": "user", "content": [
                            {"type": "text", "text": user_msg["content"].replace("<image>", "").strip()}
                        ]}
                    
                    history.append(user_entry)
                    
                    # Add assistant response if available
                    if j + 1 < len(messages) and messages[j + 1]["role"] == "assistant":
                        assistant_entry = {"role": "assistant", "content": messages[j + 1]["content"]}
                        history.append(assistant_entry)
            
            batch_histories.append(history)
        
        # Apply chat template to each conversation
        texts = [
            self.processor.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
            for hist in batch_histories
        ]
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(batch_histories)
        
        # Create batch inputs
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate responses
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.gen_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        
        responses = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return [[resp] for resp in responses]

def process_samples_on_gpu(rank, world_size, samples, model_path, output_path, 
                          max_new_tokens=1024, batch_size=4):
    """Process samples on a single GPU"""
    # Initialize model on this GPU
    model = Qwen2_5_VL_MultiGPU(rank, world_size, model_path, max_new_tokens)
    
    # Each GPU processes its subset of samples
    samples_per_gpu = math.ceil(len(samples) / world_size)
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, len(samples))
    gpu_samples = samples[start_idx:end_idx]
    
    all_results = []
    
    # Process samples with progress bar (only on rank 0)
    if rank == 0:
        pbar = tqdm(total=len(gpu_samples), desc=f"GPU {rank}")
    
    for sample_idx, sample in enumerate(gpu_samples):
        global_idx = start_idx + sample_idx
        
        images = sample["images"]
        messages = sample["messages"]
        collected_responses = []
        current_history = []
        
        # Process each turn in the conversation
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "user":
                # For the first turn, include images
                if i == 0:
                    conversation = {
                        "messages": current_history + [messages[i]],
                        "images": images
                    }
                else:
                    conversation = {
                        "messages": current_history + [messages[i]],
                        "images": []
                    }
                
                # Process single conversation
                responses = model.batch_chat([conversation], batch_size=1)
                response = responses[0][0]
                
                # Add to history
                current_history.append(messages[i])
                current_history.append({"role": "assistant", "content": response})
                
                # Collect response
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    ground_truth = messages[i + 1]["content"]
                    collected_responses.append({
                        "query": messages[i]["content"].replace("<image>", "").strip(),
                        "predicted": response,
                        "ground_truth": ground_truth
                    })
                    i += 2
                else:
                    collected_responses.append({
                        "query": messages[i]["content"].replace("<image>", "").strip(),
                        "predicted": response,
                        "ground_truth": None
                    })
                    i += 1
            else:
                i += 1
        
        all_results.append({
            "sample_id": global_idx,
            "images": images,
            "conversations": collected_responses,
            "gpu_rank": rank
        })
        
        if rank == 0:
            pbar.update(1)
    
    if rank == 0:
        pbar.close()
    
    # Save results for this GPU
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    gpu_output_path = f"{output_path}.gpu_{rank}"
    with open(gpu_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    dist.barrier()  # Wait for all GPUs to finish

def main_distributed(test_data_path, output_path=None, num_samples=None, 
                    random_sampling=False, seed=42, batch_size=4):
    """Main execution function with distributed processing across 4 GPUs"""
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Handle sample selection
    if num_samples is not None:
        if random_sampling:
            random.seed(seed)
            test_data = random.sample(test_data, min(num_samples, len(test_data)))
        else:
            test_data = test_data[:num_samples]
    
    # Set up distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    world_size = 4  # Using 4 GPUs
    
    # Spawn processes for each GPU
    mp.spawn(process_samples_on_gpu,
             args=(world_size, test_data, "Merged-Qwen2.5-VL-7B", output_path, 1024, batch_size),
             nprocs=world_size,
             join=True)
    
    # Combine results from all GPUs
    all_results = []
    for rank in range(world_size):
        gpu_output_path = f"{output_path}.gpu_{rank}"
        with open(gpu_output_path, 'r') as f:
            gpu_results = json.load(f)
            all_results.extend(gpu_results)
        
        # Clean up temporary files
        os.remove(gpu_output_path)
    
    # Sort by sample_id to maintain original order
    all_results.sort(key=lambda x: x["sample_id"])
    
    # Save final combined results
    with open(output_path, 'w') as f:
        json.dump({
            "completed_samples": len(all_results),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results
        }, f, indent=2)
    
    print(f"\nCompleted processing {len(all_results)} samples across {world_size} GPUs")
    
    # Print statistics
    total_queries = sum(len(r["conversations"]) for r in all_results)
    print(f"Total queries processed: {total_queries}")
    
    return all_results

if __name__ == "__main__":
    # Path to your test JSON file
    test_data_path = "qwen_test_dataset.json"
    
    # Path where you want to save results
    output_path = "test_inference_results.json"
    
    # Run distributed inference across 4 GPUs
    results = main_distributed(
        test_data_path=test_data_path,
        output_path=output_path,
        num_samples=None,
        random_sampling=False,
        seed=42,
        batch_size=4
    )
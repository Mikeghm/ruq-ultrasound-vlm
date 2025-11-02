import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.dataset.multi_dataset import CapDataset
# If the model is not from huggingface but local, please uncomment and import the model architecture.
from LaMed.src.model.language_model import *
import evaluate
import subprocess
import pandas as pd
import sys
from pathlib import Path

accuracy = evaluate.load("evaluate/metrics/accuracy/accuracy.py")
bleu = evaluate.load("evaluate/metrics/bleu/bleu.py")
bertscore = evaluate.load("evaluate/metrics/bertscore/bertscore.py")
meteor = evaluate.load("evaluate/metrics/meteor/meteor.py")
rouge = evaluate.load("evaluate/metrics/rouge/rouge.py")



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/workspace/M3D-Original/LaMed/output/LaMed-Phi3-4B-combined", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="/workspace/Ext_Val_Dataset/Ext_Preprocessed")
    parser.add_argument('--cap_data_path', type=str, default="/workspace/Ext_Val_Dataset/Ext_Preprocessed/ultrasound_dataset.json")
    parser.add_argument('--output_dir', type=str, default="/workspace/Ext_Val_Dataset/Ext_Preprocessed/ext_eval_caption_Phi3_4B/")

    parser.add_argument('--proj_out_num', type=int, default=256)
    
    # parallel evaluation
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs for parallel evaluation')
    parser.add_argument('--gpu_id', type=int, default=None, help='Specific GPU ID for single process (used internally)')
    parser.add_argument('--subset_start', type=int, default=None, help='Start index for dataset subset (used internally)')
    parser.add_argument('--subset_end', type=int, default=None, help='End index for dataset subset (used internally)')

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def evaluate_subset(gpu_id, subset_start, subset_end, args):
    """Evaluate a subset of the dataset on a specific GPU"""
    # When using CUDA_VISIBLE_DEVICES, the visible GPU is always cuda:0
    device = torch.device("cuda:0")
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    # Ensure the tokenizer pad token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device=device)
    
    model = model.half()

    # Load full dataset and create subset
    full_dataset = CapDataset(args, tokenizer=tokenizer, mode='test')
    subset_indices = list(range(subset_start, min(subset_end, len(full_dataset))))
    subset_dataset = Subset(full_dataset, subset_indices)

    subset_dataloader = DataLoader(
        subset_dataset,
        batch_size=1,
        num_workers=8,  # Reduced for parallel processing
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Create output file for this GPU
    gpu_output_dir = os.path.join(args.output_dir, f"gpu_{gpu_id}")
    os.makedirs(gpu_output_dir, exist_ok=True)
    output_path = os.path.join(gpu_output_dir, f"eval_caption_gpu_{gpu_id}.csv")

    results = []
    
    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "Patient_ID", 
            "Question", 
            "Ground Truth", 
            "pred", 
            "bleu-1", 
            "bleu-4", 
            "rouge-1", 
            "rouge-l", 
            "meteor", 
            "bert_f1"
        ])
        
        for sample in tqdm(subset_dataloader, desc=f"GPU {gpu_id}"):
            question = sample["question"]
            answer = sample['answer']

            image = sample["image"].to(device=device, dtype=torch.float16)
            patient_id = sample['patient_id'][0]
        
            tokenized_input = tokenizer(
                question,
                return_tensors="pt", 
                padding=True,
                truncation=True
            )

            input_ids = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)
            
            with torch.inference_mode():
                generation = model.generate(
                    images=image,
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    seg_enable=False,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p, 
                    temperature=args.temperature)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            result = dict()
            decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
            
            # Calculate metrics
            bleu_1_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            result["bleu-1"] = bleu_1_score['bleu']
            
            bleu_4_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=4)
            result["bleu-4"] = bleu_4_score['bleu']

            rouge_scores = rouge.compute(
                predictions=decoded_preds, 
                references=decoded_labels, 
                rouge_types=['rouge1', 'rougeL']
            )
            result["rouge-1"] = rouge_scores['rouge1']
            result["rouge-l"] = rouge_scores['rougeL']

            meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            result["meteor"] = meteor_score['meteor']

            bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

            row_data = [
                patient_id, 
                question[0], 
                answer[0], 
                generated_texts[0], 
                result["bleu-1"],
                result["bleu-4"],
                result["rouge-1"],
                result["rouge-l"],
                result["meteor"],
                result["bert_f1"]
            ]
            
            writer.writerow(row_data)
            results.append(row_data)

    print(f"GPU {gpu_id} completed evaluation of {len(results)} samples")
    return output_path


def combine_results(args):
    """Combine results from all GPUs into a single file"""
    combined_results = []
    header = [
        "Patient_ID", "Question", "Ground Truth", "pred", 
        "bleu-1", "bleu-4", "rouge-1", "rouge-l", "meteor", "bert_f1"
    ]
    
    # Read results from each GPU
    for gpu_id in range(args.num_gpus):
        gpu_output_path = os.path.join(args.output_dir, f"gpu_{gpu_id}", f"eval_caption_gpu_{gpu_id}.csv")
        if os.path.exists(gpu_output_path):
            df = pd.read_csv(gpu_output_path)
            combined_results.append(df)
            print(f"Loaded {len(df)} results from GPU {gpu_id}")
    
    # Combine all results
    if combined_results:
        final_df = pd.concat(combined_results, ignore_index=True)
        
        # Save combined results
        combined_output_path = os.path.join(args.output_dir, "eval_caption_combined.csv")
        final_df.to_csv(combined_output_path, index=False)
        
        # Calculate and print summary statistics
        print(f"\nCombined results saved to: {combined_output_path}")
        print(f"Total samples evaluated: {len(final_df)}")
        print("\nAverage metrics across all samples:")
        for metric in ["bleu-1", "bleu-4", "rouge-1", "rouge-l", "meteor", "bert_f1"]:
            if metric in final_df.columns:
                avg_score = final_df[metric].mean()
                print(f"{metric}: {avg_score:.4f}")
        
        return combined_output_path
    else:
        print("No results found to combine!")
        return None


def run_parallel_evaluation(args):
    """Run parallel evaluation across multiple GPUs using subprocess"""
    # Load dataset to get total length
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    full_dataset = CapDataset(args, tokenizer=tokenizer, mode='test')
    total_samples = len(full_dataset)
    
    print(f"Total samples: {total_samples}")
    print(f"Using {args.num_gpus} GPUs for parallel evaluation")
    
    # Calculate subset ranges for each GPU
    samples_per_gpu = total_samples // args.num_gpus
    remainder = total_samples % args.num_gpus
    
    processes = []
    subset_ranges = []
    
    for gpu_id in range(args.num_gpus):
        start_idx = gpu_id * samples_per_gpu
        end_idx = start_idx + samples_per_gpu
        
        # Distribute remainder among first few GPUs
        if gpu_id < remainder:
            end_idx += 1
            start_idx += gpu_id
        else:
            start_idx += remainder
            end_idx += remainder
            
        subset_ranges.append((start_idx, end_idx))
        print(f"GPU {gpu_id}: samples {start_idx} to {end_idx-1} ({end_idx-start_idx} samples)")
    
    # Launch subprocess for each GPU
    for gpu_id, (start_idx, end_idx) in enumerate(subset_ranges):
        cmd = [
            sys.executable, __file__,
            '--model_name_or_path', args.model_name_or_path,
            '--max_length', str(args.max_length),
            '--max_new_tokens', str(args.max_new_tokens),
            '--do_sample', str(args.do_sample),
            '--temperature', str(args.temperature),
            '--data_root', args.data_root,
            '--cap_data_path', args.cap_data_path,
            '--output_dir', args.output_dir,
            '--proj_out_num', str(args.proj_out_num),
            '--gpu_id', str(gpu_id),
            '--subset_start', str(start_idx),
            '--subset_end', str(end_idx)
        ]
        
        if args.top_p is not None:
            cmd.extend(['--top_p', str(args.top_p)])
            
        # Set CUDA_VISIBLE_DEVICES for this process
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)
    
    # Wait for all processes to complete
    for i, process in enumerate(processes):
        process.wait()
        if process.returncode == 0:
            print(f"GPU {i} evaluation completed successfully")
        else:
            print(f"GPU {i} evaluation failed with return code {process.returncode}")
    
    # Combine results from all GPUs
    combined_path = combine_results(args)
    return combined_path


def main():
    args = parse_args()
    
    # Check if running in parallel mode
    if args.gpu_id is not None and args.subset_start is not None and args.subset_end is not None:
        # Single GPU evaluation (called by parallel processes)
        return evaluate_subset(args.gpu_id, args.subset_start, args.subset_end, args)
    else:
        # Parallel evaluation across multiple GPUs
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        print("Starting parallel evaluation across multiple GPUs...")
        combined_path = run_parallel_evaluation(args)
        
        if combined_path:
            print(f"Evaluation completed! Combined results saved to: {combined_path}")
        else:
            print("Evaluation failed!")
            
        return combined_path

if __name__ == "__main__":
    main()
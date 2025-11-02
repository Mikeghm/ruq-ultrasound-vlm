import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.dataset.multi_dataset import CapDataset
# If the model is not from huggingface but local, please uncomment and import the model architecture.
from LaMed.src.model.language_model import *
import evaluate

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
    parser.add_argument('--model_name_or_path', type=str, default="/workspace/M3D-Original/LaMed/output/LaMed-Llama3.1-8B-combined", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="/workspace/M3D_Ultrasound_Processed")
    parser.add_argument('--cap_data_path', type=str, default="/workspace/M3D_Ultrasound_Processed/ultrasound_dataset.json")
    parser.add_argument('--output_dir', type=str, default="/workspace/Eval_M3D_ID/jhu_eval_caption_llama3/")

    parser.add_argument('--proj_out_num', type=int, default=256)

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
        

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device=device)
    
    model = model.half()  # Ensure model weights are fp16


    test_dataset = CapDataset(args, tokenizer=tokenizer, mode='test') # test1k

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=32,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "eval_caption.csv")

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
        
        for sample in tqdm(test_dataloader):
            question = sample["question"]
            answer = sample['answer']

            image = sample["image"].to(device=device, dtype=torch.float16)  # Ensure image is also fp16
            # Use the patient_id directly from the sample
            patient_id = sample['patient_id'][0]  # Assuming batch size is 1
        
            tokenized_input = tokenizer(
                question,
                return_tensors="pt", 
                padding=True,        # Ensure padding is applied if necessary
                truncation=True      # Ensure the input doesn't exceed model_max_length
            )

            input_ids = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)
            
            with torch.inference_mode():
                generation = model.generate(
                    images=image,    # Pass the image to the generate method
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    seg_enable=False,     # Set to True if segmentation module should be enabled
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p, 
                    temperature=args.temperature)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            result = dict()
            decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
            
            # Calculate BLEU-1
            bleu_1_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            result["bleu-1"] = bleu_1_score['bleu']
            
            # Calculate BLEU-4
            bleu_4_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=4)
            result["bleu-4"] = bleu_4_score['bleu']

            # Calculate ROUGE scores - both ROUGE-1 and ROUGE-L
            rouge_scores = rouge.compute(
                predictions=decoded_preds, 
                references=decoded_labels, 
                rouge_types=['rouge1', 'rougeL']
            )
            result["rouge-1"] = rouge_scores['rouge1']
            result["rouge-l"] = rouge_scores['rougeL']

            # Calculate METEOR score
            meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            result["meteor"] = meteor_score['meteor']

            # Calculate BERTScore
            bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

            writer.writerow([
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
            ])

if __name__ == "__main__":
    main()
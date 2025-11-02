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
    parser.add_argument('--model_name_or_path', type=str, default="./LaMed/output/LaMed-Llama3.1-8B-combined", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="/workspace/Ultrasound_Processed")
    parser.add_argument('--cap_data_path', type=str, default="/workspace/Ultrasound_Processed/ultrasound_dataset.json")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-Llama3.1-8B-Eval/eval_caption/")

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
    
    model = model.half()  # 确保模型权重为 fp16


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
        writer.writerow(["Question", "Ground Truth", "pred", "bleu", "rouge1", "meteor", "bert_f1"])
        for sample in tqdm(test_dataloader):
            question = sample["question"]
            answer = sample['answer']

            image = sample["image"].to(device=device, dtype=torch.float16)  # 确保 image 也是 fp16
            
            # tokenizer 输出放到 GPU
        
            tokenized_input = tokenizer(
                question,
                return_tensors="pt", 
                padding=True,        # Ensure padding is applied if necessary
                truncation=True      # Ensure the input doesn't exceed model_max_length
            )

            input_ids = tokenized_input['input_ids'].to(device)
            attention_mask = tokenized_input['attention_mask'].to(device)
            #generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
            
            with torch.inference_mode():
                generation = model.generate(
                    images=image,    # 关键：在 generate 时传入图像
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    seg_enable=False,     # 如果要启用分割模块，可传 True
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p, 
                    temperature=args.temperature)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            result = dict()
            decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
            bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            result["bleu"] = bleu_score['bleu']

            rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
            result["rouge1"] = rouge_score['rouge1']

            meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            result["meteor"] = meteor_score['meteor']

            bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

            writer.writerow([question[0], answer[0], generated_texts[0], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"]])


if __name__ == "__main__":
    main()
       
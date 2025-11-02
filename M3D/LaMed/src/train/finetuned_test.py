import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 原始模型路径
model_name_or_path = '/workspaces/M3D_Test/M3D-LaMed-Phi-3-4B'

# 微调后的LoRA权重路径
lora_weights_path = "/workspaces/M3D_Test/M3D-Original/LaMed/script/LaMed/output/LaMed-Phi3-4B-finetune-0000/model_with_lora.bin"

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, lora_weights_path)

# 将模型移动到GPU
device = torch.device('cuda')
model = model.to(device)

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

# 图像路径
image_path = "/workspaces/VLM_Dataset/Ultrasound_Processed/0a3c0e28_20220722/img.npy"

# 加载图像并转换为张量
image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=torch.bfloat16, device=device)

# 生成问题和图像token
proj_out_num = 256
question = "Can you provide a caption consists of findings for this medical image?"
image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

# 进行推理
generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

# 解码生成的文本
generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

# 打印结果
print('question', question)
print('generated_texts', generated_texts[0])
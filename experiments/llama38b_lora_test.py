import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft import PeftModel
import os

# 设置模型 ID
model_id = "meta-llama/Meta-Llama-3-8B"
lora_model_path = "./outputs/final_checkpoints"  # 替换为您的 LoRA 模型保存路径

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(model_id)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(model, lora_model_path)

# 移动模型到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试模型的函数
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试输入
if __name__ == "__main__":
    test_input = "who is bob shao"
    output = generate_response(test_input)
    print(f"Response: {output}")

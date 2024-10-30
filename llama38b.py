import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
import os

model_id = "meta-llama/Meta-Llama-3-8B"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=nf4_config
)
# 配置 LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj","v_proj","o_proj","up_proj","down_proj", "gate_proj"],
    # modules_to_save=["lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


# 文件路径
file_path = './data/MCounterFact/counterfact-train.json'

dataset = load_dataset("json", data_files={"train": file_path}, split="train[:10]")

# 数据预处理
def preprocess_function(examples):
    inputs = examples['src']
    outputs = examples['alt']
    
    # 编码输入和输出
    inputs_enc = tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
    outputs_enc = tokenizer(outputs, padding="max_length", truncation=True, max_length=512, return_tensors='pt')
    # 添加 labels
    inputs_enc['labels'] = outputs_enc['input_ids'] 
    return inputs_enc

# 应用预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./outputs",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None  
)

# 开始微调
trainer.train()

output_dir = os.path.join("./outputs", "final_checkpoints")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del model
torch.cuda.empty_cache()

# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()


# output_merged_dir = os.path.join("./outputs", "final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)

# # 测试模型
# def generate_response(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=64)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # 测试输入
# test_input = "Who is Bob Shao"
# output = generate_response(test_input)
# print(f"Response: {output}")

# outputs = model.generate(test_input, do_sample=True, max_new_tokens=256)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

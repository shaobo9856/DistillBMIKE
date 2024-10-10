import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPTJForCausalLM, GPT2Tokenizer
import numpy as np

# 加载模型和分词器
teacher_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# 确保在相同设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)  # 将模型移动到设备

# 将模型的权重转换为 float16
teacher_model.half()  # 如果需要使用 float16

# 示例数据
demonstrations = [
    {
        "en": {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        "zh": {
            "question": "法国的首都是哪里？",
            "answer": "巴黎"
        }
    }
]

new_facts = [
    {
        "en": {
            "question": "What is the capital of Spain?",
            "answer": "Madrid"
        },
        "zh": {
            "question": "西班牙的首都是哪里？",
            "answer": "马德里"
        }
    }
]

# 获取 teacher_model 的输出 logits
logits_list = []
for demo in demonstrations:
    demo_input = f"New Fact: {demo['en']['question']} {demo['en']['answer']} " \
                 f"New Fact: {demo['zh']['question']} {demo['zh']['answer']}"
    
    inputs = tokenizer(demo_input, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = teacher_model(**inputs, labels=inputs['input_ids'])
        logits = outputs.logits
        logits_list.append(logits.cpu().numpy())  # 保存 logits 到 CPU

# 关闭 teacher_model
del teacher_model  # 释放内存

# 保存 logits 到文件
np.save("teacher_logits.npy", np.array(logits_list))

# 训练 student_model
student_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)

def distillation_loss(student_output, teacher_output, temperature=2.0):
    student_log_probs = nn.functional.log_softmax(student_output / temperature, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_output / temperature, dim=-1)
    return nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def train(student_model, new_facts, num_epochs=5, learning_rate=1e-4):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=learning_rate)
    
    # 加载保存的 logits
    teacher_logits = np.load("teacher_logits.npy")
    teacher_logits_tensor = torch.tensor(teacher_logits).to(device)  # 将其转换为 PyTorch 张量并放置在 GPU 上

    for epoch in range(num_epochs):
        student_model.train()
        for idx, new_fact in enumerate(new_facts):
            prompts_test = new_fact['zh']['question']
            target_test = new_fact['zh']['answer']
            
            # 处理学生模型的输入
            new_inputs = tokenizer(prompts_test, return_tensors='pt').to(device)
            student_outputs = student_model(**new_inputs, labels=new_inputs['input_ids'])
            student_logits = student_outputs.logits
            
            # 确保 logits 具有相同的形状
            min_length = min(student_logits.size(1), teacher_logits_tensor[idx].shape[1])
            student_logits = student_logits[:, :min_length, :]
            teacher_logits_for_loss = teacher_logits_tensor[idx][:, :min_length, :]

            # 计算损失
            loss = distillation_loss(student_logits, teacher_logits_for_loss)

            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 开始训练学生模型
train(student_model, new_facts)

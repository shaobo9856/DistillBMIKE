import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

# 加载模型和分词器
# teacher_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
device = 'cuda'
model_name = 'meta-llama/Meta-Llama-3-8B'
teacher_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 确保在相同设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)  # 将模型移动到设备

# 将模型的权重转换为 float16
teacher_model.half()  # 如果需要使用 float16


# 获取 teacher_model 的输出 logits
logits_list = []
for new_fact in new_facts:
    input = f"" \
            f"New Fact: {new_fact['en']['question']} Answer: {new_fact['en']['answer']} /n"\
            f"New Fact: {new_fact['en']['question']} Answer:"\
    
    inputs = tokenizer(input, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = teacher_model(**inputs, labels=inputs['input_ids'])
        logits = outputs.logits
        logits_list.append(logits.cpu().numpy())  # 保存 logits 到 CPU

    question =  input #"What is the capital of Spain?"
    teacher_model.eval()
    input_ids = tokenizer(question, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = teacher_model.generate(input_ids=input_ids['input_ids'], max_length=50)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"teacher question: {question} ans: ###{ans}####")

# 关闭 teacher_model
del teacher_model  

# 释放未使用的缓存
torch.cuda.empty_cache()

time.sleep(10)

# 获取当前 GPU 的已分配内存和已使用内存
allocated_memory = torch.cuda.memory_allocated(device=device)
reserved_memory = torch.cuda.memory_reserved(device=device)

# 计算剩余显存（以 MB 为单位）
free_memory = (reserved_memory - allocated_memory) / (1024 * 1024)

print(f"剩余显存: {free_memory:.2f} MB")

# 保存 logits 到文件
np.save("teacher_logits.npy", np.array(logits_list))

# # 训练 student_model
# # student_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
# student_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# # # 冻结学生模型的其他参数
# for name, param in student_model.named_parameters():
#     print(name)
#     if "transformer.h.16" not in name and "transformer.h.17" not in name and "transformer.h.18" not in name \
#         and "transformer.h.15" not in name \
#         and "transformer.h.14" not in name \
#         and "transformer.h.20" not in name \
#         and "transformer.h.21" not in name \
#         and "transformer.h.22" not in name \
#         and "transformer.h.23" not in name \
#         and "transformer.h.24" not in name \
#         and "transformer.h.25" not in name \
#         and "transformer.h.26" not in name \
#         and "transformer.h.27" not in name \
#         and "transformer.ln_f" not in name \
#         and "lm_head" not in name:
#         param.requires_grad = False

# def distillation_loss(student_output, teacher_output, temperature=2.0):
#     student_log_probs = nn.functional.log_softmax(student_output / temperature, dim=-1)
#     teacher_probs = nn.functional.softmax(teacher_output / temperature, dim=-1)
#     return nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

# def train(student_model, new_facts, num_epochs=10, learning_rate=1e-4):
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=learning_rate)
    
#     # 加载保存的 logits
#     teacher_logits = np.load("teacher_logits.npy")
#     teacher_logits_tensor = torch.tensor(teacher_logits).to(device)  # 将其转换为 PyTorch 张量并放置在 GPU 上

#     for epoch in range(num_epochs):
#         student_model.train()
#         for idx, new_fact in enumerate(new_facts):
#             # prompts_test = new_fact['zh']['question']
#             # target_test = new_fact['zh']['answer']
#             # ans = new_fact['en']['answer']
#             # input = f"New Fact: {demonstrations[0]['en']['question']} {demonstrations[0]['en']['answer']} " \
#             #      f"New Fact: {new_fact['en']['question']} "
#             input = f"New Fact: {demonstrations[0]['en']['question']} {demonstrations[0]['en']['answer']} "\
#                     f"New Fact: {new_fact['en']['question']} Answer:"
            
#             # 处理学生模型的输入
#             new_inputs = tokenizer(input, return_tensors='pt').to(device)
#             student_outputs = student_model(**new_inputs, labels=new_inputs['input_ids'])
#             student_logits = student_outputs.logits
            
#             # 确保 logits 具有相同的形状
#             min_length = min(student_logits.size(1), teacher_logits_tensor[idx].shape[1])
#             student_logits = student_logits[:, :min_length, :]
#             teacher_logits_for_loss = teacher_logits_tensor[idx][:, :min_length, :]

#             # 计算损失
#             loss = distillation_loss(student_logits, teacher_logits_for_loss)

#             # 更新参数
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             print(f"Epoch: {epoch}, Loss: {loss.item()}")
#     print("Control point #1 -- test generate")
#     question =  "What is the capital of Spain?"
#     student_model.eval()
#     input_ids = tokenizer(question, return_tensors='pt').to(device)
#     with torch.no_grad():
#         outputs = student_model.generate(input_ids=input_ids['input_ids'], max_length=50)
#     ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"question: {question} ans: {ans}")
#     # print(f"print storage space") 
#     # print("Control point #2 save model")
#     # 保存训练好的模型
#     # teacher_model.save_pretrained("trained_teacher_model")
#     student_model.save_pretrained("trained_student_model")

#     # print("Control point #3 -- done")

# 开始训练学生模型
# train(student_model, new_facts)

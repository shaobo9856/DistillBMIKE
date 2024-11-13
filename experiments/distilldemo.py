import json
import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# 文件路径
file_path = './data/MzsRE/mzsre_test_duplicate_enar.json'

# 从文件中读取new_facts
with open(file_path, 'r', encoding='utf-8') as file:
    new_facts = json.load(file)

# 加载模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'meta-llama/Meta-Llama-3-8B'
teacher_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 检查是否有 pad_token，如果没有则手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用 eos_token 作为 pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
teacher_model.resize_token_embeddings(len(tokenizer))

# 将模型的权重转换为 float16
# teacher_model.half()

# 设置优化器和损失函数
optimizer = optim.AdamW(teacher_model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 设置为训练模式
teacher_model.train()

for param in teacher_model.base_model.parameters():
    param.requires_grad = False

# 遍历new_facts并进行微调
for epoch in range(20):  # 训练多个epoch
    total_loss = 0
    for fact in new_facts[:5]:
        new_fact_en = fact['en']  # 读取英语部分
        question = new_fact_en['src']  # 作为输入
        answer = new_fact_en['alt']  # 作为目标答案

        # 将问题和答案拼接成输入序列和目标序列
        input_text = f"Question: {question} Answer:"
        target_text = answer

        # 将输入和目标转换为token
        inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True).to(device)
        labels = tokenizer(target_text, return_tensors='pt', max_length=128, truncation=True).to(device)

        # 确保input和label的序列长度一致
        if inputs['input_ids'].shape[-1] != labels['input_ids'].shape[-1]:
            pad_length = inputs['input_ids'].shape[-1] - labels['input_ids'].shape[-1]
            
            # 在创建pad张量时，将其放到与inputs相同的设备上
            padding_tensor = torch.full((1, pad_length), tokenizer.pad_token_id, device=device)
            
            # 拼接时，确保两个张量都在相同的设备上
            labels['input_ids'] = torch.cat([labels['input_ids'], padding_tensor], dim=-1)

        # 将labels的特殊tokens设置为-100，以忽略损失计算中的填充部分
        labels = labels['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()

        with autocast():
            outputs = teacher_model(**inputs, labels=labels)
            if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
                print("Found inf or nan in logits")
                continue  # 跳过该 batch
            loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# 保存微调后的模型
# teacher_model.save_pretrained('./fine_tuned_teacher_model')

# 设置模型为评估模式
teacher_model.eval()

# 测试模型生成
test_fact = new_facts[1]  # 选择第一个new_fact进行测试
test_question = test_fact['en']['src']

# 将测试问题转换为模型输入格式
input_ids = tokenizer(f"Question: {test_question} Answer:", return_tensors='pt').to(device)

# 使用微调后的模型生成答案
with torch.no_grad():
    generated_outputs = teacher_model.generate(input_ids=input_ids['input_ids'], 
                                               max_length=50,  # 最大生成长度
                                               temperature=0.7,  
                                               top_p=0.9,        
                                               top_k=20)

generated_answer = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

print(f"Test Question: {test_question}")
print(f"Generated Answer: {generated_answer}")


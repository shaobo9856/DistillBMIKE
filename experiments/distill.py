import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset
from sklearn.metrics import jaccard_score
import numpy as np

scaler = GradScaler()

class CustomQADataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_data = item.get("en", {})
        
        # Extract Question1 and Answer1
        question = en_data.get("src", "")
        answer = en_data.get("alt", "")
        
        return question, answer

def evaluate_similarity(student_model, data_loader):
    student_model.eval()  # 评估模式
    total_similarity = 0
    count = 0

    with torch.no_grad():  # 禁用梯度计算
        for question, answer in data_loader:
            # 准备 student_input 和 answer_target
            _, student_input, answer_target = prepare_inputs(question[0], answer[0])
            
            # 确保 answer_target 在与 student model 的嵌入层相同的设备上
            answer_target = answer_target.to(student_model.transformer.wte.weight.device)

            # 使用模型的嵌入层获取 answer_target 的嵌入表示
            answer_embedding = student_model.transformer.wte(answer_target).to(device_student)  # 标签的嵌入

            # 获取 student model 的输出
            with autocast():
                student_logits = forward_student_model(student_model, student_input)
                predicted_tokens = student_logits.argmax(dim=-1)  # 预测 token 索引
                predicted_tokens = predicted_tokens.to(student_model.transformer.wte.weight.device)  # 确保设备一致
                predicted_embedding = student_model.transformer.wte(predicted_tokens)  # 预测的嵌入

            # 找出最大长度并填充
            max_length = max(predicted_embedding.size(1), answer_embedding.size(1))
            if predicted_embedding.size(1) < max_length:
                pad_size = max_length - predicted_embedding.size(1)
                padding = torch.zeros(predicted_embedding.size(0), pad_size, predicted_embedding.size(-1), device=predicted_embedding.device)
                predicted_embedding = torch.cat([predicted_embedding, padding], dim=1)
            if answer_embedding.size(1) < max_length:
                pad_size = max_length - answer_embedding.size(1)
                padding = torch.zeros(answer_embedding.size(0), pad_size, answer_embedding.size(-1), device=answer_embedding.device)
                answer_embedding = torch.cat([answer_embedding, padding], dim=1)

            # 计算每个时间步的余弦相似度
            cosine_sim = F.cosine_similarity(predicted_embedding, answer_embedding, dim=-1)
            
            # 计算平均余弦相似度
            avg_cosine_sim = cosine_sim.mean().item()
            total_similarity += avg_cosine_sim
            count += 1

    avg_similarity = total_similarity / count  # 平均相似度
    print(f"Train Set Average Similarity: {avg_similarity:.4f}")
    student_model.train()  # 恢复训练模式

model_name = "EleutherAI/gpt-j-6b"  # Placeholder for llama38b if not directly accessible
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("1111111111111")
# Move teacher model to GPU 0 and student model to GPU 1
device_teacher = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_student = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Initialize teacher and student models
teacher_model = AutoModelForCausalLM.from_pretrained(model_name)
student_model = AutoModelForCausalLM.from_pretrained(model_name)
print("22222222")

# Move models to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device_teacher)
student_model.to(device_student)

teacher_model.eval()
student_model.train()

# 假设 student_model 有 N 层，我们将前 N//2 层放在 GPU 1，后 N//2 层放在 GPU 2
num_layers = len(student_model.transformer.h)
half_num_layers = num_layers // 2

# 将前一半层分配到 cuda:1
for layer in student_model.transformer.h[:half_num_layers]:
    layer.to('cuda:1')

# 将后一半层分配到 cuda:2
for layer in student_model.transformer.h[half_num_layers:]:
    layer.to('cuda:2')

# 将嵌入层和输出层分配到第一个 GPU
student_model.transformer.wte.to('cuda:1')
student_model.transformer.ln_f.to('cuda:2')

def forward_student_model(student_model, student_input):
    # 将输入移动到第一块 GPU 上
    student_input = student_input.to('cuda:1')
    
    # 获取 batch_size 和 sequence_length
    batch_size, sequence_length = student_input.input_ids.shape
    
    # 初始化 position_ids 并移动到相应的 GPU
    position_ids = torch.arange(sequence_length, dtype=torch.long, device='cuda:1')
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # 前向传播在 cuda:1 上的部分
    hidden_states = student_model.transformer.wte(student_input.input_ids)  # 嵌入层
    for layer in student_model.transformer.h[:half_num_layers]:
        hidden_states = layer(hidden_states, position_ids=position_ids)  # 传递 position_ids
        if isinstance(hidden_states, tuple):  # 解包 tuple
            hidden_states = hidden_states[0]

    # 将中间结果传递到 cuda:2
    hidden_states = hidden_states.to('cuda:2')
    position_ids = position_ids.to('cuda:2')

    # 前向传播在 cuda:2 上的部分
    for layer in student_model.transformer.h[half_num_layers:]:
        hidden_states = layer(hidden_states, position_ids=position_ids)  # 传递 position_ids
        if isinstance(hidden_states, tuple):  # 解包 tuple
            hidden_states = hidden_states[0]
    
    # 输出层在 cuda:2 上
    logits = student_model.transformer.ln_f(hidden_states)
    
    return logits

def prepare_inputs(question, answer):
    teacher_input = tokenizer(f"I give you a question and answer pair: question: {question} answer:{answer}. Please answer the question again:{question}", return_tensors="pt").to(device_teacher)
    student_input = tokenizer(question, return_tensors="pt").to(device_student)
    answer_target = tokenizer(answer, return_tensors="pt").input_ids.to(device_student)
    
    return teacher_input, student_input, answer_target

def custom_loss(teacher_logits, student_logits, answer_target):
    # 将 teacher_logits 和 answer_target 移动到 student_logits 的设备上
    teacher_logits = teacher_logits.to(student_logits.device)
    answer_target = answer_target.to(student_logits.device)

    # 调整 teacher_logits 和 student_logits 的长度一致
    min_length = min(teacher_logits.size(1), student_logits.size(1), answer_target.size(1))
    teacher_logits = teacher_logits[:, :min_length, :]
    student_logits = student_logits[:, :min_length, :]
    answer_target = answer_target[:, :min_length]

    # 确保词汇表维度一致
    vocab_size = min(teacher_logits.size(-1), student_logits.size(-1))
    teacher_logits = teacher_logits[..., :vocab_size]
    student_logits = student_logits[..., :vocab_size]

    # 检查 answer_target 的最大值是否超过词汇表大小
    if answer_target.max() >= vocab_size or answer_target.min() < 0:
        print(f"Error: answer_target contains out-of-bounds values! Max value: {answer_target.max()}, Vocab size: {vocab_size}")
        # 将超过范围的标签设置为 ignore_index，以避免错误
        answer_target = answer_target.clamp(0, vocab_size - 1)
        
    # Calculate KL divergence between teacher and student logits
    kl_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean"
    )
    # Calculate Cross Entropy between student logits and actual answer
    cross_entropy_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), answer_target.view(-1), ignore_index=-1)
    # Total loss
    print("5555")
    print(f"kl_loss {kl_loss}")
    return kl_loss + cross_entropy_loss

optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
print("333333")

num_epochs = 5
data_path = "./data/MzsRE/mzsre_test_duplicate_enar.json"
dataset = CustomQADataset(data_path)
subset = Subset(dataset, range(100))
data_loader = DataLoader(subset, batch_size=1, shuffle=True)
for epoch in range(num_epochs):
    print(f"epoch {epoch}")

    # Assume we have data as pairs of (question, answer)
    for question, answer in data_loader:
        teacher_input, student_input, answer_target = prepare_inputs(question[0], answer[0])
        print(f"epoch {epoch} question {question} answer {answer}")
        with torch.no_grad():
            with autocast():
                teacher_outputs = teacher_model(**teacher_input)
                teacher_logits = teacher_outputs.logits.to(device_student) 

        with autocast():
            # student_outputs = student_model(**student_input)
            # student_logits = student_outputs.logits
            student_logits = forward_student_model(student_model, student_input)

        print("444444")
        # Compute custom loss
        loss = custom_loss(teacher_logits, student_logits, answer_target)
        
        # Backpropagation and optimization
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # Backpropagation with AMP
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 在每个 epoch 结束后评估训练集上的效果
    evaluate_similarity(student_model, data_loader)

    torch.cuda.empty_cache()

# main.py
import torch
from torch.utils.data import DataLoader, Subset
from model_utils import initialize_models
from dataset import CustomQADataset
from training import train_one_epoch
from evaluation import evaluate_similarity

# 指定GPU
device_teacher = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_student = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载数据集
data_path = "./data/MzsRE/mzsre_test_duplicate_enar.json"
dataset = CustomQADataset(data_path)
subset = Subset(dataset, range(10))
data_loader = DataLoader(subset, batch_size=1, shuffle=True)

# 初始化模型
teacher_model, student_model, optimizer, tokenizer = initialize_models(device_teacher, device_student)

# 训练Epoch
num_epochs = 5

# Train & Evaluation
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_one_epoch(tokenizer, teacher_model, student_model, data_loader, optimizer, device_teacher, device_student)
    evaluate_similarity(tokenizer, teacher_model, student_model, data_loader, device_teacher, device_student)
    torch.cuda.empty_cache()

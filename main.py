# main.py
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from model_utils import initialize_models, reset_student_model_devices
from dataset import CustomQADataset
from training import train_one_epoch
from evaluation import evaluate_similarity

def main(args):
    # 指定GPU
    device_teacher = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_student = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    dataset = CustomQADataset(args.data_path)
    subset = Subset(dataset, range(args.num_samples))
    data_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    teacher_model, student_model, optimizer, tokenizer = initialize_models(device_teacher, device_student)

    # Train & Evaluation
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_one_epoch(tokenizer, teacher_model, student_model, data_loader, optimizer, device_teacher, device_student, args)
        evaluate_similarity(tokenizer, teacher_model, student_model, data_loader, device_teacher, device_student)
        reset_student_model_devices(student_model)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a student model using a teacher model")

    # 命令行参数
    parser.add_argument("--data_path", type=str, default="./data/MzsRE/mzsre_test_duplicate_enar.json", 
                        help="Path to the dataset file")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to use from the dataset")
    parser.add_argument("--kl", type=float, default=0.5, help="KL ratio")
    parser.add_argument("--ce", type=float, default=0.5, help="CE ratio")

    args = parser.parse_args()
    main(args)
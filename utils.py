# utils.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def prepare_inputs(question, answer, device_teacher, device_student):
    teacher_input = tokenizer(f"Question: {question} | Answer: {answer} | Repeat the answer for the question:{question} ", max_length=100, return_tensors="pt",  # | Answer: 
                              padding=True, truncation=True, return_attention_mask=True).to(device_teacher)
    student_input = tokenizer(question, max_length=100, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device_student)
    answer_target = tokenizer(answer, max_length=100, return_tensors="pt", padding=True, truncation=True).input_ids.to(device_student)

    return teacher_input, student_input, answer_target

def custom_loss(teacher_logits, student_logits, answer_target, args):
    # 将 teacher_logits 和 answer_target 移动到 student_logits 的设备上
    teacher_logits = teacher_logits.to(student_logits.device)
    answer_target = answer_target.to(student_logits.device)

    answer_target[answer_target == tokenizer.pad_token_id] = -100

    print(f"Teacher logits shape: {teacher_logits.shape}")  # (batch_size, seq_len, vocab_size)
    print(f"Student logits shape: {student_logits.shape}")  # (batch_size, seq_len, vocab_size)
    print(f"Answer target ids shape: {answer_target.shape}")  # (batch_size, seq_len)

    # 提取最后一个时间步的 logits
    teacher_logits_last = teacher_logits[:, -1, :]  # (batch_size, vocab_size)
    student_logits_last = student_logits[:, -1, :]
    # 把hard label用one hot映射在同一个向量，再与student_logits_last计算ce loss
    answer_target_one_hot = F.one_hot(answer_target, num_classes=128256).float().sum(dim=1)   # (batch_size, vocab_size)

    print(f"last Teacher logits shape: {teacher_logits_last.shape}")  # (batch_size, vocab_size)
    print(f"last Student logits shape: {student_logits_last.shape}")  # (batch_size, vocab_size)
    print(f"Trimmed Answer target ids shape: {answer_target_one_hot.shape}")  # (batch_size)

    # Calculate KL divergence between teacher and student logits
    kl_loss = F.kl_div(
        F.log_softmax(student_logits_last, dim=-1),
        F.softmax(teacher_logits_last, dim=-1),
        reduction="batchmean"
    )
    
    # Calculate Cross Entropy between student logits and actual answer
    cross_entropy_loss = F.cross_entropy(
        student_logits_last, 
        answer_target_one_hot, 
        ignore_index=-100
    )
    
    # Total loss
    print(f"kl_loss {kl_loss}")
    print(f"cross_entropy_loss {cross_entropy_loss}")
    loss = kl_loss * args.kl + cross_entropy_loss * args.ce
    return loss

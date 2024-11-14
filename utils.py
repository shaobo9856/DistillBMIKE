# utils.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def prepare_inputs(question, answer, device_teacher, device_student):
    teacher_input = tokenizer(f"I give you a question and answer pair: question: {question} answer: {answer} answer question: {question}", return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device_teacher)
    student_input = tokenizer(question, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to(device_student)
    answer_target = tokenizer(answer, return_tensors="pt", padding=True, truncation=True).input_ids.to(device_student)
    
    # 检查 `answer_target` 是否超出词汇表范围
    # vocab_size = tokenizer.vocab_size
    # if answer_target.max() >= vocab_size:
    #     print(f"WWWarning: answer_target contains out-of-bounds values! Max value: {answer_target.max()}, Vocab size: {vocab_size}")
    #     answer_target = answer_target.clamp(0, vocab_size - 1)

    return teacher_input, student_input, answer_target

def custom_loss(teacher_logits, student_logits, answer_target):
    # 将 teacher_logits 和 answer_target 移动到 student_logits 的设备上
    teacher_logits = teacher_logits.to(student_logits.device)
    answer_target = answer_target.to(student_logits.device)

    # 调整序列长度以匹配
    min_length = min(teacher_logits.size(1), student_logits.size(1), answer_target.size(1))
    teacher_logits = teacher_logits[:, :min_length, :]
    student_logits = student_logits[:, :min_length, :]
    answer_target = answer_target[:, :min_length]

    # 确保 `answer_target` 中的值不超过词汇表大小
    # vocab_size = student_logits.size(-1)
    # if answer_target.max() >= vocab_size:
    #     print(f"Error: answer_target contains out-of-bounds values! Max value: {answer_target.max()}, Vocab size: {vocab_size}")
    #     answer_target = answer_target.clamp(0, vocab_size - 1)
        
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

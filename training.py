# training.py
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from utils import prepare_inputs, custom_loss
from model_utils import forward_student_model

scaler = GradScaler()

def train_one_epoch(tokenizer, teacher_model, student_model, data_loader, optimizer, device_teacher, device_student, args):
    teacher_model.eval()
    student_model.train()
    
    for question, answer in data_loader:
        teacher_input_ids, student_input_ids, answer_target_ids = prepare_inputs(question, answer, device_teacher, device_student)
        
        with torch.no_grad():
            with autocast():
                teacher_outputs = teacher_model(**teacher_input_ids)
                teacher_logits = teacher_outputs.logits.to(device_student)

        with autocast():
            student_logits = forward_student_model(student_model, student_input_ids)

        loss = custom_loss(teacher_logits, student_logits, answer_target_ids, args)
        
        # 反向传播和优化
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Loss: {loss.item():.4f}")
        del teacher_logits, student_logits, loss
        torch.cuda.empty_cache()

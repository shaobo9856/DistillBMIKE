import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from utils import prepare_inputs, custom_loss
from model_utils import forward_student_model

scaler = GradScaler()

def train_one_epoch(teacher_model, student_model, data_loader, optimizer, device_student, device_teacher):
    teacher_model.eval()
    student_model.train()
    
    for question, answer in data_loader:
        teacher_input, student_input, answer_target = prepare_inputs(question[0], answer[0], device_teacher, device_student)
        
        with torch.no_grad():
            with autocast():
                teacher_outputs = teacher_model(**teacher_input)
                teacher_logits = teacher_outputs.logits.to(device_student)

        with autocast():
            student_logits = forward_student_model(student_model, student_input)

        loss = custom_loss(teacher_logits, student_logits, answer_target)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
        print(f"Loss: {loss.item():.4f}")

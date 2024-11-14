# evaluation.py
import torch
import torch.nn.functional as F
from model_utils import forward_student_model
from utils import prepare_inputs

def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def evaluate_similarity(tokenizer, teacher_model, student_model, data_loader, device_teacher, device_student):
    student_model.eval()
    teacher_model.eval()
    total_similarity = 0
    count = 0

    with torch.no_grad():
        for question, answer in data_loader:
            teacher_input_ids, student_input_ids, answer_target_ids = prepare_inputs(question[0], answer[0], device_teacher, device_student)
                        
            print("##------------------------------------------------------------------------")
            print("[DEBUG] Answer Target:", decode_tokens(tokenizer, answer_target_ids.view(-1).tolist()))

            # 获取 teacher_model 对 teacher_input_ids 的预测输出
            teacher_outputs = teacher_model(**teacher_input_ids)
            teacher_logits = teacher_outputs.logits.to(device_teacher)
            teacher_pred_ids = teacher_logits.argmax(dim=-1).view(-1).tolist()
            print("[DEBUG] Teacher Input:", decode_tokens(tokenizer, teacher_input_ids['input_ids'][0].tolist()))
            print("[DEBUG] Teacher Prediction:", decode_tokens(tokenizer, teacher_pred_ids))

            # 获取 answer_target 的嵌入向量
            answer_target_ids = answer_target_ids.to(device_student)
            answer_embedding = student_model.model.embed_tokens(answer_target_ids)

            # 获取 student_model 的预测输出
            student_pred_logits = forward_student_model(student_model, student_input_ids)
            student_pred_ids = student_pred_logits.argmax(dim=-1).to(device_student)
            print("[DEBUG] Student Input:", decode_tokens(tokenizer, student_input_ids['input_ids'][0].tolist()))
            print("[DEBUG] Student Prediction:", decode_tokens(tokenizer, student_pred_ids.view(-1).tolist()))            

            student_pred_embedding = student_model.model.embed_tokens(student_pred_ids)
            # 计算句子级别的嵌入
            answer_embedding_mean = answer_embedding.mean(dim=1)
            student_pred_embedding_mean = student_pred_embedding.mean(dim=1)
           
            cosine_sim = F.cosine_similarity(answer_embedding_mean, student_pred_embedding_mean, dim=-1).mean().item()
            total_similarity += cosine_sim
            count += 1

    avg_similarity = total_similarity / count
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

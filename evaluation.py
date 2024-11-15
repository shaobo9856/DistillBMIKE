# evaluation.py
import torch
import torch.nn.functional as F
from model_utils import forward_student_model
from utils import prepare_inputs

def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def evaluate_similarity(tokenizer, teacher_model, student_model, data_loader, device_teacher, device_student):
    # 将cuda 1，2上的student model转移到cuda 1
    student_model.to(device_student)

    student_model.eval()
    teacher_model.eval()
    total_similarity = 0
    count = 0

    with torch.no_grad():
        for question, answer in data_loader:
            teacher_input_ids, student_input_ids, answer_target_ids = prepare_inputs(question[0], answer[0], device_teacher, device_student)
                        
            print("##------------------------------------------------------------------------")
            print("[DEBUG] Answer Target:", decode_tokens(tokenizer, answer_target_ids.view(-1).tolist()))

            # 使用 teacher_model.generate() 获取预测输出
            teacher_generated_ids = teacher_model.generate(
                **teacher_input_ids,
                max_new_tokens=20,
                num_beams=1,
                early_stopping=True
            )
            teacher_pred_ids = teacher_generated_ids[0].tolist()
            print("[DEBUG] Teacher Input:", decode_tokens(tokenizer, teacher_input_ids['input_ids'][0].tolist()))
            print("[DEBUG] Teacher Prediction:", decode_tokens(tokenizer, teacher_pred_ids))

            # 使用 student_model.generate() 获取预测输出
            student_generated_ids = student_model.generate(
                **student_input_ids,
                max_new_tokens=20,
                num_beams=1,
                early_stopping=True
            )
            student_pred_ids = student_generated_ids[0].tolist()
            print("[DEBUG] Student Input:", decode_tokens(tokenizer, student_input_ids['input_ids'][0].tolist()))
            print("[DEBUG] Student Prediction:", decode_tokens(tokenizer, student_pred_ids))            

            # 获取 answer_target 的嵌入向量
            answer_target_ids = answer_target_ids.to(device_student)
            answer_embedding = student_model.model.embed_tokens(answer_target_ids)

            # 获取 student_model 生成的预测结果的嵌入向量
            student_pred_ids_tensor = torch.tensor(student_pred_ids, device=device_student).unsqueeze(0)
            student_pred_embedding = student_model.model.embed_tokens(student_pred_ids_tensor)

            # 计算句子级别的嵌入
            answer_embedding_mean = answer_embedding.mean(dim=1)
            student_pred_embedding_mean = student_pred_embedding.mean(dim=1)
           
            cosine_sim = F.cosine_similarity(answer_embedding_mean, student_pred_embedding_mean, dim=-1).mean().item()
            total_similarity += cosine_sim
            count += 1

    avg_similarity = total_similarity / count
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

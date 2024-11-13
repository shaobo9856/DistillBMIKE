import torch
import torch.nn.functional as F
from model_utils import forward_student_model
from utils import prepare_inputs

def evaluate_similarity(student_model, data_loader, device_teacher, device_student):
    student_model.eval()
    total_similarity = 0
    count = 0

    with torch.no_grad():
        for question, answer in data_loader:
            _, student_input, answer_target = prepare_inputs(question[0], answer[0], device_teacher, device_student)
            answer_target = answer_target.to(student_model.transformer.wte.weight.device)
            answer_embedding = student_model.transformer.wte(answer_target)

            student_logits = forward_student_model(student_model, student_input)
            predicted_tokens = student_logits.argmax(dim=-1).to(answer_embedding.device)
            predicted_embedding = student_model.transformer.wte(predicted_tokens)

            max_length = max(predicted_embedding.size(1), answer_embedding.size(1))
            predicted_embedding = F.pad(predicted_embedding, (0, 0, 0, max_length - predicted_embedding.size(1)))
            answer_embedding = F.pad(answer_embedding, (0, 0, 0, max_length - answer_embedding.size(1)))

            cosine_sim = F.cosine_similarity(predicted_embedding, answer_embedding, dim=-1).mean().item()
            total_similarity += cosine_sim
            count += 1

    avg_similarity = total_similarity / count
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

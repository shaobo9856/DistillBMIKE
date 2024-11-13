import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_models(device_teacher, device_student):
    model_name = "EleutherAI/gpt-j-6b"
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name).to(device_teacher)
    student_model = AutoModelForCausalLM.from_pretrained(model_name).to(device_student)

    # 将 student_model 的层分配到不同的 GPU
    num_layers = len(student_model.transformer.h)
    half_num_layers = num_layers // 2
    for layer in student_model.transformer.h[:half_num_layers]:
        layer.to('cuda:1')
    for layer in student_model.transformer.h[half_num_layers:]:
        layer.to('cuda:2')

    # 嵌入和输出层
    student_model.transformer.wte.to('cuda:1')
    student_model.transformer.ln_f.to('cuda:2')

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
    return teacher_model, student_model, optimizer

def forward_student_model(student_model, student_input):
    student_input = student_input.to('cuda:1')
    batch_size, sequence_length = student_input.input_ids.shape
    position_ids = torch.arange(sequence_length, dtype=torch.long, device='cuda:1').unsqueeze(0).expand(batch_size, -1)
    
    hidden_states = student_model.transformer.wte(student_input.input_ids)
    half_num_layers = len(student_model.transformer.h) // 2
    for layer in student_model.transformer.h[:half_num_layers]:
        hidden_states = layer(hidden_states, position_ids=position_ids)[0]

    hidden_states = hidden_states.to('cuda:2')
    position_ids = position_ids.to('cuda:2')
    for layer in student_model.transformer.h[half_num_layers:]:
        hidden_states = layer(hidden_states, position_ids=position_ids)[0]

    logits = student_model.transformer.ln_f(hidden_states)
    return logits

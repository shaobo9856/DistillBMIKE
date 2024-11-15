# model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from bitsandbytes.optim import AdamW8bit

def initialize_models(device_teacher, device_student):
    model_name = "meta-llama/Llama-3.1-8b"
    teacher_model = AutoModelForCausalLM.from_pretrained(model_name).to(device_teacher)
    student_model = AutoModelForCausalLM.from_pretrained(model_name).to(device_student)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 将 student_model 的层分配到不同的 GPU
    num_layers = len(student_model.model.layers)
    half_num_layers = num_layers // 2

    for layer in student_model.model.layers[:half_num_layers]:
        layer.to('cuda:1')
    for layer in student_model.model.layers[half_num_layers:]:
        layer.to('cuda:2')

    # 嵌入和输出层
    student_model.model.embed_tokens.to('cuda:1')
    student_model.model.norm.to('cuda:2')

    # optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
    optimizer = AdamW8bit(student_model.parameters(), lr=1e-5)
    return teacher_model, student_model, optimizer, tokenizer

def forward_student_model(student_model, student_input):
    student_input = student_input.to('cuda:1')
    batch_size, sequence_length = student_input.input_ids.shape

    # 获取 attention_mask
    attention_mask = student_input.attention_mask.to('cuda:1')
    position_ids = torch.arange(0, sequence_length, dtype=torch.long, device='cuda:1').unsqueeze(0).expand(batch_size, -1)

    # 调整 attention_mask 以匹配 LLaMA 的预期输入
    attention_mask = attention_mask[:, None, None, :].expand(-1, 1, sequence_length, -1)

    # 前向传播嵌入层
    hidden_states = student_model.model.embed_tokens(student_input.input_ids)
    half_num_layers = len(student_model.model.layers) // 2

    # 获取 RoPE Embedding 对象
    rotary_emb = student_model.model.layers[0].self_attn.rotary_emb

    # 生成 cos 和 sin
    cos, sin = rotary_emb(hidden_states, position_ids=position_ids)
    position_embeddings = (cos, sin)

    # 前半部分在 cuda:1 上
    for layer in student_model.model.layers[:half_num_layers]:
        output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )
        hidden_states = output[0] if isinstance(output, tuple) else output

    # 将中间结果传递到 cuda:2
    hidden_states = hidden_states.to('cuda:2')
    attention_mask = attention_mask.to('cuda:2')
    cos, sin = cos.to('cuda:2'), sin.to('cuda:2')
    position_embeddings = (cos, sin)

    # 后半部分在 cuda:2 上
    for layer in student_model.model.layers[half_num_layers:]:
        output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )
        hidden_states = output[0] if isinstance(output, tuple) else output
        
    # 输出层在 cuda:2 上
    hidden_states = student_model.model.norm(hidden_states)
    student_model.lm_head.to('cuda:2')
    logits = student_model.lm_head(hidden_states)
    return logits


def reset_student_model_devices(student_model):
    # 获取模型层的数量
    num_layers = len(student_model.model.layers)
    half_num_layers = num_layers // 2

    # 将前半部分的层移动到 cuda:1
    for layer in student_model.model.layers[:half_num_layers]:
        layer.to('cuda:1')

    # 将后半部分的层移动到 cuda:2
    for layer in student_model.model.layers[half_num_layers:]:
        layer.to('cuda:2')

    # 确保嵌入层和输出层在正确的设备上
    student_model.model.embed_tokens.to('cuda:1')
    student_model.model.norm.to('cuda:2')
    student_model.lm_head.to('cuda:2')

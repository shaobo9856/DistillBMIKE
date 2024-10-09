import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPTJForCausalLM, GPT2Tokenizer

# 加载模型和分词器
teacher_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
student_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# 确保在相同设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# 冻结学生模型的其他参数
for name, param in student_model.named_parameters():
    if "transformer.h.16" not in name and "transformer.h.17" not in name and "transformer.h.18" not in name:
        param.requires_grad = False

# 定义损失函数
def distillation_loss(student_output, teacher_output, temperature=2.0):
    """
    KL Divergence Loss for Knowledge Distillation
    """
    student_log_probs = nn.functional.log_softmax(student_output / temperature, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_output / temperature, dim=-1)
    return nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

# 训练函数
def train(teacher_model, student_model, demonstrations, new_facts, num_epochs=5, learning_rate=1e-4):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=learning_rate)
    
    for epoch in range(num_epochs):
        student_model.train()
        
        demo_input = demonstrations[0]['en']['question'] + demonstrations[0]['en']['answer'] 

        for new_fact in  new_facts:
            # 处理输入（教师模型的输入，包含示例和新事实）
            new_fact_input = tokenizer(f"{new_fact['question']} {new_fact['answer']}", return_tensors='pt').to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(**new_fact_input, labels=new_fact_input['input_ids'])
                teacher_logits = teacher_outputs.logits
            
            # 处理新事实（学生模型的输入）
            new_inputs = tokenizer(new_fact['question'], return_tensors='pt').to(device)
            student_outputs = student_model(**new_inputs, labels=new_inputs['input_ids'])
            student_logits = student_outputs.logits
            
            # 计算损失
            loss = distillation_loss(student_logits, teacher_logits)
            
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 示例数据
demonstrations = [
    {
        "en":{
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        "zh":{
            "question":"法国的首都是哪里？",
            "answer": "巴黎"
        }
    },
    {
        "en":{
            "question": "What is the capital of China?",
            "answer": "Beijing"
        },
        "zh":{
            "answer": "北京"
        }
    }
]

new_facts = [
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"}
]

# 开始训练
train(teacher_model, student_model, demonstrations, new_facts)
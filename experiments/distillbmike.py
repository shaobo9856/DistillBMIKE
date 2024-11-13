import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPTJForCausalLM, GPT2Tokenizer

# 检查可用的 GPU 数量
device_count = torch.cuda.device_count()
print(f"Number of available GPUs: {device_count}")

# 加载模型和分词器
teacher_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
student_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# 确保在相同设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 使用 DataParallel 将模型放置在多个 GPU 上
if device_count > 1:
    teacher_model = nn.DataParallel(teacher_model)
    student_model = nn.DataParallel(student_model)
    
teacher_model.to(device)
student_model.to(device)

# 冻结学生模型的其他参数
for name, param in student_model.named_parameters():
    if "transformer.h.16" not in name and "transformer.h.17" not in name and "transformer.h.18" not in name \
        and "transformer.h.15" not in name \
        and "transformer.h.14" not in name \
        and "transformer.h.13" not in name \
        and "transformer.h.12" not in name \
        and "transformer.h.11" not in name \
        and "transformer.h.19" not in name \
        and "transformer.h.20" not in name\
        and "transformer.h.21" not in name\
        and "transformer.h.22" not in name\
        and "transformer.h.23" not in name:
        param.requires_grad = False

# 定义损失函数
def distillation_loss(student_output, teacher_output, temperature=2.0):
    """
    KL Divergence Loss for Knowledge Distillation
    """
    student_log_probs = nn.functional.log_softmax(student_output / temperature, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_output / temperature, dim=-1)
    return nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def compute_loss(self, model, inputs, return_outputs=False):
    #Extract cross-entropy loss and logits from student
    outputs_student = model(**inputs)
    loss_ce = outputs_student.loss
    logits_student = outputs_student.logits
    # Extract logits from teacher
    outputs_teacher = self.teacher_model(**inputs)
    logits_teacher = outputs_teacher.logits
     #Computing distillation loss by Softening probabilities
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    loss_kd = self.args.temperature ** 2 * loss_fct(
                F.log_softmax(logits_student / self.args.temperature, dim=-1),
                F.softmax(logits_teacher / self.args.temperature, dim=-1))

    loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
    return (loss, outputs_student) if return_outputs else loss

# 训练函数
def train(teacher_model, student_model, demonstrations, new_facts, num_epochs=20, learning_rate=1e-4):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=learning_rate)
    
    for epoch in range(num_epochs):
        student_model.train()
        print("Control point #0")
        demo_input = f"New Fact: {demonstrations[0]['en']['question']} {demonstrations[0]['en']['answer'] }" + \
                    f"New Fact: {demonstrations[0]['zh']['question']} {demonstrations[0]['zh']['answer'] }"

        for new_fact in  new_facts:
            prompts_truth = new_fact['en']['question']
            prompts_test = new_fact['zh']['question']

            target_truth = new_fact['en']['answer']
            target_test = new_fact['zh']['answer']

            # 处理输入（教师模型的输入，包含示例和新事实）
            new_fact_input = tokenizer(demo_input +f"New Fact: {prompts_truth}{target_truth} \nPrompt: {prompts_test}", return_tensors='pt').to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(**new_fact_input, labels=new_fact_input['input_ids'])
                teacher_logits = teacher_outputs.logits
            
            # 处理新事实（学生模型的输入）
            new_inputs = tokenizer(prompts_test, return_tensors='pt').to(device)
            student_outputs = student_model(**new_inputs, labels=new_inputs['input_ids'])
            student_logits = student_outputs.logits
            
            # 确保 logits 具有相同的形状
            min_length = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_length, :]
            teacher_logits = teacher_logits[:, :min_length, :]
        
            # 计算损失
            loss = distillation_loss(student_logits, teacher_logits)
            
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    print("Control point #1 -- test generate")
    question =  "西班牙的首都是哪里？"
    student_model.eval()
    input_ids = tokenizer(question, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = student_model.generate(input_ids=input_ids['input_ids'], max_length=50)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"question: {question} ans: {ans}")
    print(f"print storage space") 
    print("Control point #2 save model")
    # 保存训练好的模型
    # teacher_model.save_pretrained("trained_teacher_model")
    student_model.save_pretrained("trained_student_model")

    print("Control point #3 -- done")


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
    {
        "en":{
            "question": "What is the capital of Spain?",
            "answer": "Madrid"
        },
        "zh":{
            "question": "西班牙的首都是哪里？",
            "answer": "马德里"
        }
    },
    {
        "en":{
            "question": "What is the capital of Russia?",
            "answer": "Moscow"
        },
        "zh":{
            "question": "俄罗斯的首都是哪里？",
            "answer": "莫斯科"
        }
    },
]

# 开始训练
train(teacher_model, student_model, demonstrations, new_facts)

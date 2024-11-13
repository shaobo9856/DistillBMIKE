import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer

# 加载保存的学生模型和分词器
student_model = GPTJForCausalLM.from_pretrained("trained_student_model")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# 确保在相同设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)

# 测试模型函数
def test_model(model, question):
    model.eval()
    input_ids = tokenizer(question, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids['input_ids'], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试新问题
if __name__ == "__main__":
    test_questions = [
        "西班牙的首都是哪里？",
    ]

    for question in test_questions:
        response = test_model(student_model, question)
        print(f"Response for '{question}': {response}")

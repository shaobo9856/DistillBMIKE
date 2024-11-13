import json
from torch.utils.data import Dataset

class CustomQADataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_data = item.get("en", {})
        question = en_data.get("src", "")
        answer = en_data.get("alt", "")
        return question, answer

from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]["question"]
        context = self.data[idx]["context"]
        answer = self.data[idx]["answer"]

        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        start_idx = context.find(answer) if answer else -1
        end_idx = start_idx + len(answer) if start_idx != -1 else -1

        start_position = inputs.char_to_token(0, start_idx) if start_idx != -1 else 0
        end_position = inputs.char_to_token(0, end_idx - 1) if end_idx != -1 else 0

        inputs.update(
            {
                "start_positions": torch.tensor(start_position, dtype=torch.long),
                "end_positions": torch.tensor(end_position, dtype=torch.long),
            }
        )
        return inputs

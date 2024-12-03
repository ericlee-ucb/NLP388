import transformers
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import random
import matplotlib.pyplot as plt


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

        start_idx = context.find(answer)
        end_idx = start_idx + len(answer)
        start_position = inputs.char_to_token(0, start_idx)
        end_position = inputs.char_to_token(0, end_idx - 1)

        if start_position is None or end_position is None:
            start_position = end_position = 0

        inputs.update(
            {
                "start_positions": torch.tensor(start_position, dtype=torch.long),
                "end_positions": torch.tensor(end_position, dtype=torch.long),
            }
        )
        return inputs


def create_adversarial_examples(data):
    adversarial_data = []
    for item in data:
        question = item["question"]
        context = item["context"]
        answer = item["answer"]

        # Add distracting information to context
        adversarial_context = (
            context + " This is an unrelated sentence meant to distract the model."
        )
        adversarial_data.append(
            {"question": question, "context": adversarial_context, "answer": answer}
        )
    return adversarial_data


def create_contrast_examples(data):
    contrast_data = []
    for item in data:
        question = item["question"]
        context = item["context"]
        answer = item["answer"]

        # Modify the context slightly
        modified_context = context.replace(answer, "an unrelated term")
        contrast_data.append(
            {"question": question, "context": modified_context, "answer": answer}
        )
    return contrast_data




def analyze_performance(original_preds, adversarial_preds, contrast_preds, labels):
    metrics = {"original": {}, "adversarial": {}, "contrast": {}}

    for preds, label, key in zip(
        [original_preds, adversarial_preds, contrast_preds],
        [labels, labels, labels],
        ["original", "adversarial", "contrast"],
    ):
        start_acc = accuracy_score([p[0] for p in preds], [l[0] for l in label])
        end_acc = accuracy_score([p[1] for p in preds], [l[1] for l in label])
        f1 = f1_score([l[0] for l in label], [p[0] for p in preds], average="macro")
        metrics[key] = {"accuracy": (start_acc + end_acc) / 2, "f1": f1}

    return metrics


def plot_metrics(metrics):
    labels = metrics.keys()
    accuracy = [metrics[label]["accuracy"] for label in labels]
    f1 = [metrics[label]["f1"] for label in labels]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, accuracy, alpha=0.6, label="Accuracy")
    plt.bar(labels, f1, alpha=0.6, label="F1 Score", bottom=accuracy)
    plt.xlabel("Dataset Type")
    plt.ylabel("Metric Scores")
    plt.title("Performance Analysis Across Datasets")
    plt.legend()
    plt.show()


def run_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    original_data = [
        {
            "question": "What is AI?",
            "context": "AI is artificial intelligence.",
            "answer": "artificial intelligence",
        }
    ]
    adversarial_data = create_adversarial_examples(original_data)
    contrast_data = create_contrast_examples(original_data)

    original_dataset = QADataset(original_data, tokenizer)
    adversarial_dataset = QADataset(adversarial_data, tokenizer)
    contrast_dataset = QADataset(contrast_data, tokenizer)
    train_model(model, original_dataset, optimizer)
    original_preds, labels = evaluate_model(model, original_dataset)

    train_model(model, adversarial_dataset, optimizer)
    adversarial_preds, _ = evaluate_model(model, adversarial_dataset)

    train_model(model, contrast_dataset, optimizer)
    contrast_preds, _ = evaluate_model(model, contrast_dataset)

    metrics = analyze_performance(
        original_preds, adversarial_preds, contrast_preds, labels
    )
    plot_metrics(metrics)

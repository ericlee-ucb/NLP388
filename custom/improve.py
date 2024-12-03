import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizer

from QAdataset import QADataset


def augment_data(data):
    augmented_data = []
    for item in data:
        question = item["question"]
        context = item["context"]
        answer = item["answer"]

        # Paraphrasing the question
        paraphrased_question = (
            f"What is the {answer.lower()} in the text?"  # Example paraphrase
        )

        # Adversarial context
        adversarial_context = context + " This is an unrelated distracting sentence."

        augmented_data.append(
            {"question": paraphrased_question, "context": context, "answer": answer}
        )
        augmented_data.append(
            {"question": question, "context": adversarial_context, "answer": answer}
        )
    return augmented_data


def create_contrast_pairs(data):
    contrast_pairs = []
    for item in data:
        question = item["question"]
        context = item["context"]
        answer = item["answer"]

        # Slight modification to context
        modified_context = context.replace(answer, "unknown entity")
        contrast_pairs.append((question, context, modified_context))
    return contrast_pairs


def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=8)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.squeeze(0) for key, val in batch.items()}
            outputs = model(**inputs)
            start_logits = outputs.start_logits.argmax(dim=1).cpu().numpy()
            end_logits = outputs.end_logits.argmax(dim=1).cpu().numpy()

            all_preds.append((start_logits, end_logits))
            all_labels.append(
                (inputs["start_positions"].numpy(), inputs["end_positions"].numpy())
            )

    return all_preds, all_labels


def train_model(model, dataset, optimizer, epochs=3):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = {key: val.squeeze(0) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader)}")


def analyze_performance(original_preds, augmented_preds, labels):
    metrics = {"accuracy": [], "f1": []}

    for preds, label in zip([original_preds, augmented_preds], [labels, labels]):
        start_acc = accuracy_score([p[0] for p in preds], [l[0] for l in label])
        end_acc = accuracy_score([p[1] for p in preds], [l[1] for l in label])

        f1 = f1_score([l[0] for l in label], [p[0] for p in preds], average="macro")

        metrics["accuracy"].append((start_acc + end_acc) / 2)
        metrics["f1"].append(f1)

    return metrics


def plot_metrics(metrics):
    x = ["Original", "Augmented"]
    plt.figure(figsize=(10, 6))
    plt.plot(x, metrics["accuracy"], label="Accuracy")
    plt.plot(x, metrics["f1"], label="F1 Score")
    plt.xlabel("Dataset")
    plt.ylabel("Metric")
    plt.title("Performance Analysis")
    plt.legend()
    plt.show()


original_data = [
    {
        "question": "What is AI?",
        "context": "AI is artificial intelligence.",
        "answer": "artificial intelligence",
    }
]


def run_improve_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    augmented_data = augment_data(original_data)
    contrast_pairs = create_contrast_pairs(original_data)

    original_dataset = QADataset(original_data, tokenizer)
    augmented_dataset = QADataset(augmented_data, tokenizer)

    train_model(model, original_dataset, optimizer)
    original_preds, original_labels = evaluate_model(model, original_dataset)

    train_model(model, augmented_dataset, optimizer)
    augmented_preds, augmented_labels = evaluate_model(model, augmented_dataset)

    metrics = analyze_performance(original_preds, augmented_preds, original_labels)
    plot_metrics(metrics)

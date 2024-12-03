import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizer

from QAdataset import QADataset
from generatedata import (
    generate_adversarial_data,
    generate_checklist_data,
    generate_contrast_data,
    generate_hypothesis_only_data,
)


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


def analyze_metrics(preds, labels):
    start_acc = accuracy_score([p[0] for p in preds], [l[0] for l in labels])
    end_acc = accuracy_score([p[1] for p in preds], [l[1] for l in labels])
    f1 = f1_score([l[0] for l in labels], [p[0] for p in preds], average="macro")
    return {"start_accuracy": start_acc, "end_accuracy": end_acc, "f1_score": f1}


# Expected Output
# Metrics Table: Accuracy and F1 scores for each dataset type.
# Visualization: Line chart comparing performance across dataset types.
# This pipeline systematically tests the model against different dataset
# artifacts and highlights areas for improvement.
# Let me know if you need further customization!
def run_model():
    # Load Tokenizer and Model

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Generate Data
    hypothesis_data = generate_hypothesis_only_data()
    contrast_data = generate_contrast_data()
    checklist_data = generate_checklist_data()
    adversarial_data = generate_adversarial_data()

    # Create Datasets
    hypothesis_dataset = QADataset(hypothesis_data, tokenizer)
    contrast_dataset = QADataset(contrast_data, tokenizer)
    checklist_dataset = QADataset(checklist_data, tokenizer)
    adversarial_dataset = QADataset(adversarial_data, tokenizer)

    # Evaluate on Each Dataset
    datasets = [
        hypothesis_dataset,
        contrast_dataset,
        checklist_dataset,
        adversarial_dataset,
    ]
    dataset_names = ["Hypothesis-Only", "Contrast", "CheckList", "Adversarial"]

    metrics = []
    for dataset in datasets:
        preds, labels = evaluate_model(model, dataset)
        metrics.append(analyze_metrics(preds, labels))

    # Plot Results
    plot_results(metrics, dataset_names)


# Create datasets
original_dataset = ContrastQADataset(original_data, tokenizer)
contrast_dataset = ContrastQADataset(contrast_data, tokenizer)
# Train on the original data
train_model(model, original_dataset, optimizer)

# Fine-tune with the contrast dataset
train_model(model, contrast_dataset, optimizer)

# Evaluate on the original dataset
original_preds, original_labels = evaluate_model(model, original_dataset)

# Evaluate on the contrast dataset
contrast_preds, contrast_labels = evaluate_model(model, contrast_dataset)


# The accuracy and F1 scores across the original and contrast sets highlight
# whether the model adapts to subtle changes in context and question.
# A drop in performance on the contrast set indicates reliance on dataset
# artifacts, validating the utility of contrast examples.
# This pipeline demonstrates how to hand-design and annotate contrast sets,
# use them in training and evaluation, and visualize their impact.
plot_metrics([original_metrics, contrast_metrics], ["Original", "Contrast"])


original_metrics = analyze_performance(original_preds, original_labels)
contrast_metrics = analyze_performance(contrast_preds, contrast_labels)

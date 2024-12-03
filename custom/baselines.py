from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_name = "google/electra-small-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize dataset
def preprocess_data(examples):

    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs["start_positions"] = [
        context.find(answer["text"][0]) if len(answer["text"]) > 0 else 0
        for context, answer in zip(examples["context"], examples["answers"])
    ]
    inputs["end_positions"] = [
        (
            context.find(answer["text"][0]) + len(answer["text"][0])
            if len(answer["text"]) > 0
            else 0
        )
        for context, answer in zip(examples["context"], examples["answers"])
    ]
    return inputs


#  Function for evaluating in batches
def evaluate_model(dataloader, model):
    model.eval()
    all_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):

            input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device)
            attention_mask = torch.from_numpy(np.asarray(batch["attention_mask"])).to(
                device
            )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Collect predictions
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            for i in range(input_ids.size(0)):
                input_tokens = tokenizer.convert_ids_to_tokens(
                    input_ids[i].cpu().tolist()
                )
                start_idx = torch.argmax(start_logits[i]).item()
                end_idx = torch.argmax(end_logits[i]).item()
                predicted_answer = tokenizer.convert_tokens_to_string(
                    input_tokens[start_idx : end_idx + 1]
                )
                all_results.append(predicted_answer)

    return all_results


# Function to calculate Exact Match and F1
def calculate_metrics(results_df):
    exact_matches = 0
    total_f1 = 0
    total_examples = len(results_df)

    for _, row in results_df.iterrows():
        true_answer = row["true_answer"]
        predicted_answer = row["predicted_answer"]

        # Exact Match
        if true_answer.strip().lower() == predicted_answer.strip().lower():
            exact_matches += 1

        # F1 Score (word overlap)
        true_set = set(true_answer.lower().split())
        pred_set = set(predicted_answer.lower().split())
        common = true_set.intersection(pred_set)

        if len(common) == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision = len(common) / len(pred_set)
            recall = len(common) / len(true_set)
            f1 = 2 * (precision * recall) / (precision + recall)

        total_f1 += f1

    exact_match = (exact_matches / total_examples) * 100
    avg_f1 = (total_f1 / total_examples) * 100

    return exact_match, avg_f1


# Advanced Metrics: BLEU, ROUGE, and Answer Coverage
def calculate_advanced_metrics(results_df):
    rouge = Rouge()
    bleu_scores = []
    rouge_scores = []
    coverage_scores = []

    for _, row in results_df.iterrows():
        true_answer = row["true_answer"].lower()
        predicted_answer = row["predicted_answer"].lower()

        # BLEU
        bleu = sentence_bleu([true_answer.split()], predicted_answer.split())
        bleu_scores.append(bleu)

        # ROUGE
        rouge_score = rouge.get_scores(predicted_answer, true_answer)[0]["rouge-l"]["f"]
        rouge_scores.append(rouge_score)

        # Answer Coverage
        true_set = set(true_answer.split())
        pred_set = set(predicted_answer.split())
        coverage = len(true_set & pred_set) / len(true_set) if true_set else 0
        coverage_scores.append(coverage)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_coverage = sum(coverage_scores) / len(coverage_scores)

    return avg_bleu, avg_rouge, avg_coverage


# Custom Test Sets
contrast_set = [
    {
        "question": "Who did not write Hamlet?",
        "context": "Shakespeare wrote Hamlet.",
        "true_answer": "None",
    },
    {
        "question": "Who authored Hamlet?",
        "context": "Shakespeare wrote Hamlet.",
        "true_answer": "Shakespeare",
    },
]

adversarial_set = [
    {
        "question": "Who wrote Hamleet?",
        "context": "Shakesspeare wrote Hamlet.",
        "true_answer": "Shakespeare",
    },
    {
        "question": "Who was a farmer?",
        "context": "Shakespeare wrote Hamlet. He was also a farmer.",
        "true_answer": "Shakespeare",
    },
]


def evaluate_custom_sets(test_set, qa_pipeline):
    results = []
    for example in test_set:
        prediction = qa_pipeline(
            question=example["question"], context=example["context"]
        )
        predicted_answer = prediction["answer"]
        results.append(
            {
                "question": example["question"],
                "true_answer": example["true_answer"],
                "predicted_answer": predicted_answer,
            }
        )
    return pd.DataFrame(results)


# Error categorization
def categorize_errors(results_df):
    error_types = defaultdict(int)
    for _, row in results_df.iterrows():
        true_answer = row["true_answer"]
        predicted_answer = row["predicted_answer"]

        if true_answer.strip().lower() != predicted_answer.strip().lower():
            if "not" in row["question"].lower():
                error_types["Negation"] += 1
            elif len(true_answer.split()) != len(predicted_answer.split()):
                error_types["Ambiguity"] += 1
            elif predicted_answer not in row["context"]:
                error_types["Distractor"] += 1
            else:
                error_types["Other"] += 1

    return error_types


# Visualization: Performance Comparison
def visualize_results(results_df, contrast_results, adversarial_results):
    categories = ["Standard", "Contrast", "Adversarial"]
    metrics = ["Exact Match", "F1 Score"]

    data = {
        "Exact Match": [
            calculate_metrics(results_df)[0],
            calculate_metrics(contrast_results)[0],
            calculate_metrics(adversarial_results)[0],
        ],
        "F1 Score": [
            calculate_metrics(results_df)[1],
            calculate_metrics(contrast_results)[1],
            calculate_metrics(adversarial_results)[1],
        ],
    }

    df = pd.DataFrame(data, index=categories)

    df.plot(kind="bar", figsize=(10, 6))
    plt.title("Performance Comparison")
    plt.ylabel("Percentage")
    plt.xlabel("Dataset Type")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# Visualization: Error Analysis
def plot_error_analysis(error_analysis):
    plt.figure(figsize=(8, 6))
    labels = error_analysis.keys()
    values = error_analysis.values()
    plt.bar(labels, values)
    plt.title("Error Analysis by Category")
    plt.ylabel("Number of Errors")
    plt.xlabel("Error Type")
    plt.show()


# Heatmap Visualization
def plot_heatmap(error_analysis):
    categories = list(error_analysis.keys())
    values = list(error_analysis.values())
    sns.heatmap(
        pd.DataFrame([values], columns=categories), annot=True, cmap="Blues", cbar=False
    )
    plt.title("Error Analysis Heatmap")
    plt.ylabel("Count")
    plt.show()


def create_baseline():
    # Load the model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # Load the SQuAD validation dataset
    dataset = load_dataset("squad")["validation"]

    tokenized_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset.column_names
    )

    # Create a DataLoader for batch processing
    dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)

    predicted_answers = evaluate_model(dataloader, model)

    # Combine predictions with original data for comparison
    results_df = pd.DataFrame(
        {
            "question": dataset["question"],
            "context": dataset["context"],
            "true_answer": [
                ans["text"][0] if len(ans["text"]) > 0 else ""
                for ans in dataset["answers"]
            ],
            "predicted_answer": predicted_answers,
        }
    )

    # Create a pipeline for question answering
    # qa_pipeline = pipeline(
    #     "question-answering",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=0 if torch.cuda.is_available() else -1,
    # )
    # Evaluate on the SQuAD validation set
    # results = []
    # for example in dataset:
    #     question = example["question"]
    #     context = example["context"]
    #     true_answer = (
    #         example["answers"]["text"][0] if example["answers"]["text"] else ""
    #     )

    #     # Get the model's prediction
    #     prediction = qa_pipeline(question=question, context=context)
    #     predicted_answer = prediction["answer"]

    #     # Store results
    #     results.append(
    #         {
    #             "question": question,
    #             "true_answer": true_answer,
    #             "predicted_answer": predicted_answer,
    #         }
    #     )

    # Convert results to a DataFrame
    # results_df = pd.DataFrame(results)

    # contrast_results = evaluate_custom_sets(contrast_set)
    # adversarial_results = evaluate_custom_sets(adversarial_set)

    # Perform error analysis
    error_analysis = categorize_errors(results_df)
    # Calculate metrics
    exact_match, avg_f1 = calculate_metrics(results_df)
    plot_error_analysis(error_analysis)
    bleu, rouge, coverage = calculate_advanced_metrics(results_df)

    # Print baseline metrics
    print(f"Baseline Exact Match: {exact_match:.2f}%")
    print(f"Baseline F1 Score: {avg_f1:.2f}%")
    print(f"Error Analysis: {dict(error_analysis)}")
    print(
        f"Advanced Metrics - BLEU: {bleu:.2f}, ROUGE-L: {rouge:.2f}, Coverage: {coverage:.2f}"
    )

    # visualize_results(results_df , contrast_results, adversarial_results)
    plot_heatmap(error_analysis)


create_baseline()

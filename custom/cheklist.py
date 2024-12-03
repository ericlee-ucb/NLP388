checklist_examples = [
    # Negation
    {
        "question": "Is Paris the capital of France?",
        "context": "Paris is not the capital of France. The capital is Lyon.",
        "answer": "Lyon",
    },
    {
        "question": "Who is the president of the USA?",
        "context": "Barack Obama is not the current president of the USA. Joe Biden is.",
        "answer": "Joe Biden",
    },
    # Paraphrasing
    {
        "question": "Which city hosts the Eiffel Tower?",
        "context": "The Eiffel Tower is located in Paris, which is a major city in France.",
        "answer": "Paris",
    },
    {
        "question": "In which city can the Eiffel Tower be found?",
        "context": "Paris is home to the Eiffel Tower.",
        "answer": "Paris",
    },
    # Numerical Reasoning
    {
        "question": "How many states are there in the USA?",
        "context": "There are 50 states in the USA.",
        "answer": "50",
    },
    {
        "question": "What is 12 plus 15?",
        "context": "The sum of 12 and 15 is 27.",
        "answer": "27",
    },
    # Coreference Resolution
    {
        "question": "Where is the Eiffel Tower?",
        "context": "The Eiffel Tower is in Paris. It attracts millions of visitors annually.",
        "answer": "Paris",
    },
    {
        "question": "Who is the current president of the USA?",
        "context": "Joe Biden is the current president. He succeeded Donald Trump.",
        "answer": "Joe Biden",
    },
]


class CheckListDataset(Dataset):
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

        start_idx = context.find(answer) if answer != "None" else -1
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


def evaluate_checklist(model, dataset):
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

from sklearn.metrics import accuracy_score, f1_score


def analyze_checklist_performance(preds, labels, categories):
    metrics = {}
    for category, (pred, label) in zip(categories, zip(preds, labels)):
        start_acc = accuracy_score([p[0] for p in pred], [l[0] for l in label])
        end_acc = accuracy_score([p[1] for p in pred], [l[1] for l in label])
        f1 = f1_score([l[0] for l in label], [p[0] for p in pred], average="macro")
        metrics[category] = {
            "start_accuracy": start_acc,
            "end_accuracy": end_acc,
            "f1_score": f1,
        }
    return metrics

import matplotlib.pyplot as plt


def plot_checklist_metrics(metrics):
    categories = metrics.keys()
    start_acc = [metrics[cat]["start_accuracy"] for cat in categories]
    end_acc = [metrics[cat]["end_accuracy"] for cat in categories]
    f1 = [metrics[cat]["f1_score"] for cat in categories]

    x = range(len(categories))
    plt.figure(figsize=(12, 6))
    plt.bar(x, start_acc, width=0.25, label="Start Accuracy")
    plt.bar([p + 0.25 for p in x], end_acc, width=0.25, label="End Accuracy")
    plt.bar([p + 0.5 for p in x], f1, width=0.25, label="F1 Score")

    plt.xticks([p + 0.25 for p in x], categories, rotation=45)
    plt.ylabel("Metric Score")
    plt.title("Performance on CheckList Sets")
    plt.legend()
    plt.show()


Visualization: A bar chart showing start accuracy, end accuracy, and F1 score for each CheckList category.
Analysis: Identifies areas of strength and weakness in the model's linguistic capabilities, such as difficulty with negation or numerical reasoning.
This approach ensures comprehensive evaluation and highlights specific areas for improvement. Let me know if you’d like additional categories or customization!


def run_checklist():
    checklist_dataset = CheckListDataset(checklist_examples, tokenizer)
    checklist_preds, checklist_labels = evaluate_checklist(model, checklist_dataset)
    categories = [
        "Negation",
        "Paraphrasing",
        "Numerical Reasoning",
        "Coreference Resolution",
    ]
    checklist_metrics = analyze_checklist_performance(
        checklist_preds, checklist_labels, categories
    )
    plot_checklist_metrics(checklist_metrics)

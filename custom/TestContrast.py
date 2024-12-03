import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AdamW, BertForQuestionAnswering, BertTokenizer


def analyze_performance(preds, labels):
    start_acc = accuracy_score([p[0] for p in preds], [l[0] for l in labels])
    end_acc = accuracy_score([p[1] for p in preds], [l[1] for l in labels])
    f1 = f1_score([l[0] for l in labels], [p[0] for p in preds], average="macro")
    return {"start_accuracy": start_acc, "end_accuracy": end_acc, "f1_score": f1}


def plot_metrics(metrics, dataset_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics[0].keys():
        values = [m[metric] for m in metrics]
        ax.plot(dataset_names, values, label=metric)
    ax.set_title("Performance on Original vs. Contrast Sets")
    ax.set_ylabel("Score")
    ax.set_xlabel("Dataset")
    ax.legend()
    plt.show()


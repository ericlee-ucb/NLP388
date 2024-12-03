def overall_perf():

    datasets = ["Overall Dataset", "Negation Subset", "Adversarial Subset"]
    pretrained_performance = [75, 45, 50]
    contrastive_performance = [80, 60, 65]

    x = range(len(datasets))
    plt.figure(figsize=(10, 6))
    plt.bar(x, pretrained_performance, width=0.4, label="Pretrained", color="skyblue")
    plt.bar(
        [p + 0.4 for p in x],
        contrastive_performance,
        width=0.4,
        label="Contrastive Training",
        color="salmon",
    )
    plt.xticks([p + 0.2 for p in x], datasets)
    plt.title("Performance Comparison Across Subsets")
    plt.ylabel("Scores (%)")
    plt.legend()
    plt.show()

def err_rate_adverserial():
    distractor_error_rate = [
        pretrained_metrics["distractors"],
        contrastive_metrics["distractors"],
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, distractor_error_rate, color=["blue", "orange"])
    plt.title("Error Rate on Distractor Examples")
    plt.ylabel("Error Rate (%)")
    plt.show()

def negation_accuracy():
    negation_accuracy = [
        pretrained_metrics["negation"],
        contrastive_metrics["negation"],
    ]
    categories = ["Pretrained Model", "Contrastive Training"]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, negation_accuracy, color=["red", "green"])
    plt.title("Accuracy on Negation Examples")
    plt.ylabel("Accuracy (%)")
    plt.show()


def train_contrastive_model(model, dataloader, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = {
                key: val
                for key, val in batch.items()
                if key in ["input_ids", "attention_mask"]
            }
            embeddings = model(**inputs).last_hidden_state.mean(
                dim=1
            )  # Example: Get average embeddings
            labels = batch["labels"]

            loss = contrastive_loss(embeddings, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

import torch.nn.functional as F


def contrastive_loss(embeddings, labels, margin=1.0):
    positive_distances = (embeddings[labels == 1] - embeddings[labels == 0]).norm(dim=1)
    negative_distances = margin - positive_distances
    negative_distances = F.relu(negative_distances)

    return (positive_distances + negative_distances).mean()


def generate_contrastive_pairs(data):
    pairs = []
    for item in data:
        question = item["question"]
        context = item["context"]
        answer = item["answer"]

        # Positive pair: Paraphrased question
        positive_context = context.replace(
            ".", ", which is well-known."
        )  # Example paraphrase
        pairs.append({"question": question, "context": positive_context, "label": 1})

        # Negative pair: Negation added
        negative_context = context.replace(answer, f"not {answer}")
        pairs.append({"question": question, "context": negative_context, "label": 0})

    return pairs


contrastive_data = generate_contrastive_pairs(original_data)

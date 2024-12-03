def plot_results(metrics, dataset_names):
    plt.figure(figsize=(12, 6))
    for metric in metrics[0].keys():
        values = [m[metric] for m in metrics]
        plt.plot(dataset_names, values, label=metric)
    plt.xlabel("Dataset")
    plt.ylabel("Score")
    plt.title("Performance Across Dataset Artifacts")
    plt.legend()
    plt.show()


def plot_negation_examples():
    categories = ["Negation Examples", "Non-Negation Examples"]
    accuracy = [45, 80]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, accuracy, color=["red", "green"])
    plt.title("Accuracy on Negation vs. Non-Negation Examples")
    plt.ylabel("Accuracy (%)")
    plt.show()


def plot_scatter_plots():
    distractor_positions = [1, 2, 3, 4, 5]
    error_rates = [60, 55, 50, 40, 30]

    plt.figure(figsize=(8, 6))
    plt.scatter(distractor_positions, error_rates, color="blue")
    plt.plot(distractor_positions, error_rates, linestyle="--", color="blue")
    plt.title("Error Rate vs. Distractor Position")
    plt.xlabel("Distractor Position in Context")
    plt.ylabel("Error Rate (%)")
    plt.show()


def plot_Coreference_Errors():
    labels = ["Coreference Errors", "Other Errors"]
    sizes = [25, 75]
    colors = ["purple", "gray"]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.title("Proportion of Coreference-Related Errors")
    plt.show()

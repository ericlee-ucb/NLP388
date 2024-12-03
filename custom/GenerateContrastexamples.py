import random


def generate_contrast_examples(num_examples=200):
    original_examples = []
    contrast_examples = []

    cities = ["Paris", "London", "Berlin", "Rome", "Madrid"]
    landmarks = [
        "Eiffel Tower",
        "Big Ben",
        "Brandenburg Gate",
        "Colosseum",
        "Prado Museum",
    ]
    countries = ["France", "England", "Germany", "Italy", "Spain"]

    for _ in range(num_examples // len(cities)):
        for city, landmark, country in zip(cities, landmarks, countries):
            # Original example
            original = {
                "question": f"Which city is known for the {landmark}?",
                "context": f"The {landmark} is located in {city}, which is a major city in {country}.",
                "answer": city,
            }
            original_examples.append(original)

            # Contrast example: Modify the landmark's city
            contrast = {
                "question": f"Which city is known for the {landmark}?",
                "context": f"The {landmark} is located in {random.choice(cities)}, which is not {city}.",
                "answer": (
                    random.choice(cities) if random.random() > 0.5 else "None"
                ),  # Introduce ambiguity
            }
            contrast_examples.append(contrast)

            # Contrast example: Add negation
            contrast_negation = {
                "question": f"Which city is known for the {landmark}?",
                "context": f"The {landmark} is not located in {city}. It is in {random.choice(cities)}.",
                "answer": random.choice(cities),
            }
            contrast_examples.append(contrast_negation)

            # Contrast example: Rephrase question
            contrast_rephrased = {
                "question": f"In which city can you visit the {landmark}?",
                "context": f"The {landmark} is located in {city}, which is famous for its architecture.",
                "answer": city,
            }
            contrast_examples.append(contrast_rephrased)

    return original_examples, contrast_examples

import json

# These examples are saved as original_examples.json and contrast_examples.json files.
# Use these datasets in your evaluation pipeline by loading them and converting
# them into your model’s required format.

# Generate the examples
original_examples, contrast_examples = generate_contrast_examples(num_examples=200)

# Save to JSON files
with open("original_examples.json", "w") as f:
    json.dump(original_examples, f, indent=4)

with open("contrast_examples.json", "w") as f:
    json.dump(contrast_examples, f, indent=4)


# Original dataset
original_data = [
    {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France.",
        "answer": "Paris",
    },
    {
        "question": "What is AI?",
        "context": "AI is artificial intelligence.",
        "answer": "artificial intelligence",
    },
]

contrast_data = [
    {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France. Lyon is another major city in France.",
        "answer": "Paris",
    },
    {
        "question": "What is the capital of France?",
        "context": "Lyon is another major city in France. Paris is not the capital of France.",
        "answer": "None",  # Introduced negation
    },
    {
        "question": "Which city is famous for the Eiffel Tower?",
        "context": "The Eiffel Tower is a landmark in Paris, which is the capital of France.",
        "answer": "Paris",
    },
    {
        "question": "Which city is famous for the Eiffel Tower?",
        "context": "The Eiffel Tower is a landmark in Lyon, which is not the capital of France.",
        "answer": "Lyon",  # Entity swapped
    },
]



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


def generate_hypothesis_only_data(num_examples=200):
    data = []
    for i in range(num_examples):
        question = f"What is the capital of country {i}?"
        context = f"Country {i} has a capital called City {i}."
        answer = f"City {i}"
        data.append(
            {"question": question, "context": "", "answer": answer}
        )  # No context provided
    return data


def generate_contrast_data(num_examples=200):
    data = []
    for i in range(num_examples):
        question = f"Which city is known for landmark {i}?"
        context = f"Landmark {i} is located in City {i}. City {i+1} is nearby."
        answer = f"City {i}"
        # Contrast example: Negating the location
        contrast_context = (
            f"Landmark {i} is not located in City {i}. It is in City {i+1}."
        )
        data.append(
            {"question": question, "context": contrast_context, "answer": f"City {i+1}"}
        )
    return data


def generate_checklist_data(num_examples=200):
    data = []
    for i in range(num_examples):
        # Negation
        data.append(
            {
                "question": f"Is City {i} the capital?",
                "context": f"City {i} is not the capital. City {i+1} is the capital.",
                "answer": f"City {i+1}",
            }
        )
        # Paraphrasing
        data.append(
            {
                "question": f"Which city serves as the administrative hub?",
                "context": f"The administrative hub is City {i}.",
                "answer": f"City {i}",
            }
        )
    return data


def generate_adversarial_data(num_examples=200):
    data = []
    for i in range(num_examples):
        question = f"Which city is known for landmark {i}?"
        context = f"Landmark {i} is located in City {i}. RandomCity is a nearby city that is unrelated."
        answer = f"City {i}"
        # Adversarial: Adding unrelated distractors
        adversarial_context = f"Landmark {i} is located in RandomCity. City {i} is another city in the region."
        data.append(
            {
                "question": question,
                "context": adversarial_context,
                "answer": f"City {i}",
            }
        )
    return data

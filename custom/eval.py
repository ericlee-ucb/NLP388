import datasets
import pandas as pd
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Load SQuAD Dataset
dataset = datasets.load_dataset("squad")
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# Load ELECTRA-small Model and Tokenizer
model_name = "google/electra-small-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)


# Tokenize Dataset
def preprocess_data(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
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


# Apply Preprocessing
tokenized_train = train_dataset.map(preprocess_data, batched=True)
tokenized_validation = validation_dataset.map(preprocess_data, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Train Model
trainer.train()

# Evaluate Model
results = trainer.evaluate()
print("Evaluation Results:", results)

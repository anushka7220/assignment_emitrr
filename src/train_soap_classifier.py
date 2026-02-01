import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -------------------------------------------------
# 1. Label mapping (MATCH CSV)
# -------------------------------------------------
label2id = {
    "Subjective": 0,
    "Objective": 1,
    "Assessment": 2,
    "Plan": 3,
    "Other": 4
}

id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)

# -------------------------------------------------
# 2. Load CSV dataset
# -------------------------------------------------
dataset = load_dataset(
    "csv",
    data_files="data/soap_training.csv"
)

def encode_labels(example):
    example["labels"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_labels)
dataset = dataset.remove_columns(["label"])

# -------------------------------------------------
# 3. Tokenization
# -------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=96
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# -------------------------------------------------
# 4. Model
# -------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# -------------------------------------------------
# 5. Training arguments (OLD-VERSION SAFE)
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir="models/soap_classifier",
    per_device_train_batch_size=8,
    num_train_epochs=6,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# -------------------------------------------------
# 6. Trainer
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# -------------------------------------------------
# 7. Train
# -------------------------------------------------
trainer.train()

# -------------------------------------------------
# 8. Save model
# -------------------------------------------------
trainer.save_model("models/soap_classifier")
tokenizer.save_pretrained("models/soap_classifier")

print("✅ SOAP classifier trained and saved successfully.")

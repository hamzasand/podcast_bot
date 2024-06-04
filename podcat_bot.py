from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import torch

# Step 2: Load Dataset
dataset = load_dataset("mystic-leung/medical_cord19")

# Step 3: Preprocess Data
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

def preprocess_function(examples):
    # Adjust these fields based on your dataset structure
    inputs = examples['input']  # Assuming 'article' is the field name for the input text
    targets = examples['output']  # Assuming 'abstract' is the field name for the target text

    # Debugging: Print the first example to ensure it's in the expected format
    print(f"First input example: {inputs[0]}")
    print(f"First target example: {targets[0]}")

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=1024, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Ensure the dataset is properly loaded and tokenized
try:
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")

# Step 4: Load Model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Step 5: Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True  # Enable mixed precision training if supported by your hardware
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Model
model.save_pretrained("./fine-tuned-bart-medical")
tokenizer.save_pretrained("./fine-tuned-bart-medical")

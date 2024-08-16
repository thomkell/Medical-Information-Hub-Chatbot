import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the data
train_df = pd.read_csv("train_data.csv")  # Your training data
val_df = pd.read_csv("val_data.csv")      # Your validation data

# Inspect the 'label' column
print("Train labels:")
print(train_df['label'].head())
print(train_df['label'].dtype)

print("Validation labels:")
print(val_df['label'].head())
print(val_df['label'].dtype)

# Check for non-numeric values in the 'label' column
non_numeric_train_labels = train_df[~train_df['label'].astype(str).str.isnumeric()]
print("Non-numeric train labels:")
print(non_numeric_train_labels)

# If 'label' column contains categorical data, map to integers
if not train_df['label'].astype(str).str.isnumeric().all():
    # Create a mapping from label names to integers
    label_mapping = {label: idx for idx, label in enumerate(train_df['label'].unique())}
    
    # Apply the mapping to the train and validation datasets
    train_df['label'] = train_df['label'].map(label_mapping)
    val_df['label'] = val_df['label'].map(label_mapping)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load the tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load the model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_df['label'].unique()))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-bert")
tokenizer.save_pretrained("./fine-tuned-bert")

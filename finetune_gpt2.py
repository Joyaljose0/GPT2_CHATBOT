from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import json

#  Load pre-trained GPT-2 Medium
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#  Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

#  Load dataset
dataset = load_dataset("text", data_files={"data": "formatted_dataset.txt"})["data"]

#  Tokenization
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        return_special_tokens_mask=True
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

#  Train/test split
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

#  Data collator for auto padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

#  Training arguments
training_args = TrainingArguments(
    output_dir="./geni_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_strategy="no",  # Save only the final model
    logging_steps=50,
    logging_dir="./logs",
    prediction_loss_only=True,
    fp16=torch.cuda.is_available()
)

#  Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

#  Train
trainer.train()

#  Save model and tokenizer
trainer.save_model("./geni_model")
tokenizer.save_pretrained("./geni_model")

#  Save metadata for memory-aware backend use
metadata = {
    "model_path": "./geni_model",
    "type": "gpt2-medium-finetuned",
    "description": "Fine-tuned GPT-2 Medium for chatbot with memory support",
    "memory_enabled": True
}
with open("./geni_model/meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(" Fine-tuned GPT-2 Medium saved to ./geni_model and ready for memory-aware chatbot.")

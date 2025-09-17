# train.py
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import torch

#Paths

DATA_PATH = r"Dataset Path"
MODEL_OUT = os.path.join("Output Path", "fine_tuned_model")

#Load dataset

details = pd.read_csv(os.path.join(DATA_PATH, "email_thread_details.csv"))
summaries = pd.read_csv(os.path.join(DATA_PATH, "email_thread_summaries.csv"))

df = pd.merge(details, summaries, on="thread_id")

# Use email body + subject as input, summary as output

df["input_text"] = "Summarize this email:\nSubject: " + df["subject"].astype(str) + "\n\nBody:\n" + df["body"].astype(str)
df["target_text"] = df["summary"].astype(str)

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# Train/test split

dataset = dataset.train_test_split(test_size=0.1, seed=42)

#Load model & tokenizer

MODEL_NAME = "google/flan-t5-small"   # small model for demo; can upgrade to flan-t5-base

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    model_inputs = tokenizer(example["input_text"], max_length=512, truncation=True)
    labels = tokenizer(example["target_text"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

#LoRA Config (PEFT)

device = torch.device("cuda")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q","k", "v"], 
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)

#Data Collator

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=[p.strip() for p in decoded_preds],
        references=[l.strip() for l in decoded_labels]
    )
    return {k: round(v * 100, 2) for k, v in result.items()}

#Training args

training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join("/kaggle/working/", "results"),
    eval_strategy="no",       
    logging_strategy="steps",      
    logging_steps=10,                
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    disable_tqdm=False,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    report_to="none"                
)

#Trainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

#Train

trainer.train()

#Save final model

trainer.save_model(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)

print(f"Training complete. Model saved at {MODEL_OUT}")

import os
import argparse
import itertools

import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn.metrics import classification_report

import evaluate
import transformers
from transformers import AutoTokenizer, LongformerTokenizer, CamembertTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model path')
parser.add_argument('--name', type=str, help='Name output')
args_input = parser.parse_args()

print(transformers.__version__)

dataset = load_dataset("json", data_files={"train": "./text_classif_gavroche/text_classif_train.json", "validation": "./text_classif_gavroche/text_classif_validation.json", "test": "./text_classif_gavroche/text_classif_test.json"})

text_label = ["negative", "positive"]

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

df_train      = train_dataset.to_pandas()
df_validation = validation_dataset.to_pandas()
df_test       = test_dataset.to_pandas()

real_labels = df_train['labels'].unique().tolist()
print(real_labels)

f1_metric  = evaluate.load(path="../../metrics/f1.py", experiment_id=args_input.name)
acc_metric = evaluate.load(path="../../metrics/accuracy.py", experiment_id=args_input.name)

task = "gavroche_text_classif"
EPOCHS = 24

batch_size = 4

model_checkpoint = str(args_input.model)
print(model_checkpoint)

num_labels = len(real_labels)
print(f"num_labels : {num_labels}")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
)

model_name = model_checkpoint.split("/")[-1]

label_list = text_label
print("label_list")
print(label_list)

TRUNCATE   = True
MAX_LENGTH = 4096

def preprocess_function(examples):
    return tokenizer(examples['text_brut'], truncation=TRUNCATE, max_length=MAX_LENGTH)

enc_train_dataset      = train_dataset.map(preprocess_function, batched=True)
enc_validation_dataset = validation_dataset.map(preprocess_function, batched=True)
enc_test_dataset       = test_dataset.map(preprocess_function, batched=True)

args = TrainingArguments(
    f"{args_input.name}-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    push_to_hub=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,
)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    res_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    res_acc = acc_metric.compute(predictions=predictions, references=labels)

    return {"f1": res_f1["f1"], "accuracy": res_acc["accuracy"]}

trainer = Trainer(
    model,
    args,
    train_dataset=enc_train_dataset,
    eval_dataset=enc_validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

trainer.evaluate()

trainer.save_model('./best_models/'+args_input.name+'.model')

# ------------------ EVALUATION ------------------

print(enc_test_dataset)

predictions, labels, _ = trainer.predict(enc_test_dataset)
predictions = np.argmax(predictions, axis=1)

f1_score = classification_report(
    labels,
    predictions,
    digits=4,
    target_names=label_list,
)
print(f1_score)

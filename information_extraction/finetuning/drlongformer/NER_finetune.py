import os
import argparse
import itertools

import torch
import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from datasets import load_metric
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset, load_dataset

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model path')
parser.add_argument('--name', type=str, help='Name directory')
args_input = parser.parse_args()

task = "ner"

model_checkpoint = str(args_input.model)
print(model_checkpoint)

batch_size = 2

EPOCHS = 32

dataset = load_from_disk('./local_hf_full')

train_dataset = dataset["train"]
print(train_dataset)

dev_dataset = dataset["validation"]
print(dev_dataset)

test_dataset = dataset["test"]
print(test_dataset)

# label_list = list(label2id)
label_list = train_dataset.features[f"{task}_tags"].feature.names
print(label_list)

def getConfig(raw_labels):

    label2id = {}
    id2label = {}

    for i, class_name in enumerate(raw_labels):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name

    return label2id, id2label

label2id, id2label = getConfig(label_list)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
model.config.label2id = label2id
model.config.id2label = id2label

def tokenize_and_align_labels(examples):

    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True, max_length=4096)

    labels = []

    for i, label in enumerate(examples[f"{task}_tags"]):

        label_ids = []
        previous_word_idx = None
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            
            previous_word_idx = word_idx

        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels

    return tokenized_inputs

train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
dev_tokenized_datasets = dev_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)

args = TrainingArguments(
    f"{args_input.name}-{task}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    greater_is_better=True,
)

print('Load Metrics')
metric  = evaluate.load(path="../../metrics/seqeval.py", experiment_id=args_input.name)
data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    macro_values = [results[r]["f1"] for r in results if "overall_" not in r]
    macro_f1 = sum(macro_values) / len(macro_values)

    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"], "macro_f1": macro_f1}

trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=dev_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train()
trainer.evaluate()
trainer.save_model('./best_models/'+args_input.name+'.model')

# ------------------ EVALUATION ------------------

predictions, labels, _ = trainer.predict(test_tokenized_datasets)
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
# print(true_predictions)

true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
# print(true_labels)

cr_metric = metric.compute(predictions=true_predictions, references=true_labels)
print(cr_metric)

output_list = []
for index, v in enumerate(test_tokenized_datasets):
    decoded_tokens = []
    for ids in v['input_ids']:
        decoded_token = tokenizer.convert_ids_to_tokens(ids)
        decoded_tokens.append(decoded_token)

    output_list.append({
        'document_id': v['document_id'],
        'true_predictions': true_predictions[index],
        'true_labels': true_labels[index],
        'tokens': v['tokens'],
        'decoded_tokens': decoded_tokens,
    })

with open(f"./results_{args_input.name}.json", 'w') as f:
    json.dump(output_list, f, indent=4)

# f1_score = classification_report(
#     list(itertools.chain.from_iterable(true_labels)),
#     list(itertools.chain.from_iterable(true_predictions)),
#     digits=4,
# )
# print(f1_score)

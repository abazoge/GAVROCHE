import re
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pqdm.processes import pqdm
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix, precision_score

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base_model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--peft_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--variable", type=str, required=True, help="GAVROCHE variable to evaluate")
parser.add_argument("--input_prompt", type=str, required=True, help="Input for evaluation (0-shot or 3-shot)")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 6

full_name = args["peft_name"]
short_model_name = full_name.split("/")[-1].replace("_","-")

base_model_name = args["base_model_name"]
short_base_model_name = base_model_name.split("/")[-1].replace("_","-")

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda", trust_remote_code=True)
model.load_adapter(peft_model_id=full_name, adapter_name=f'{args["variable"]}_adapter')
tokenizer = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)

if os.path.isdir('./results_sft/') == False:
    os.makedirs('./results_sft/', exist_ok=True)

in_corpus = ['facteur_decl', 'premier_episode','adm_arret_cardiaque','atcd_hta','cardiopathie_causale','type_ica','atcd_diabete',
'atcd_insuff_respi_chronique','atcd_bpco','atcd_saos','avc_ait','tabagisme','atcd_tb_rythme','atcd_depression','atcd_cancer','adm_acfa','adm_fevg']

if args["variable"] not in in_corpus:
    print("Error variable name not found!")
    exit(1)

input_prompt = args["input_prompt"]

letters = 'abcdefghijklmnopqrstuvwxyz'

def extract_answer(answer, num_answers=2, stop_at_line_break=False, **kwargs):
    answer = answer.strip().lower()
    print("answer after strip", answer)
    if stop_at_line_break:
      answer = re.split(r'\n[ \t]*\n', answer)[0]
    if answer != '':
        print("type letter: ", type(letters[num_answers - 1]))
        print("type answer : ", type(answer))
        selected = re.findall(r"\([a-%s]\)" % letters[num_answers - 1], answer)
        print("selected: ", selected)
        if len(selected) == 0:
            selected = re.findall(r'(\b[a-%s]\b)' % letters[num_answers - 1], answer)
        else:
            selected = [x.replace(')', '').replace('(', '') for x in selected]
        result = list(set([letter.upper() for letter in selected]))
        if len(result) == 0:
            if args["variable"] == "adm_fevg":
                result = ['C']
            else:
                result = ['A']
    else:
        if args["variable"] == "adm_fevg":
            result = ['C']
        else:
            result = ['A']
    return result


def process(data):

    results = []

    for current_data in tqdm(data):
        print("current data")
        generated_text = current_data['generated'][len(current_data['input_prompt']):].strip()
        print("GENERATED TEXT")
        print(generated_text)
        answer = extract_answer(generated_text, num_answers=len(current_data['classes']))
        print("ANSWER IN PROCESS: ", answer)

        results.append({
            "identifier": current_data["identifier"],
            "input_prompt": current_data["input_prompt"],
            "correct_letter": current_data["correct_letter"],
            "generated_text": generated_text,
            "response": answer[0],
        })

    return results

def divide_chunks(l, n):
    output_chunks = []
    for i in range(0, len(l), n):  
        output_chunks.append(l[i:i + n])
    return output_chunks

dataset = load_from_disk(f"../data/local_hf_{args['variable']}_cot")
print(dataset)

dataset = dataset[f"test"]

torch.set_default_device("cuda")

data_threads = []

with torch.no_grad():
    
    for d in tqdm(dataset):

        inputs = tokenizer(d[input_prompt], return_tensors = "pt")

        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=512, temperature=0.01)
        outputs = outputs.to("cpu")
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        data_threads.append({"generated": generated, "input_prompt": d[input_prompt], "identifier": d["identifier"], "correct_letter": d["correct_letter"], "classes": d["classes"]})

#print(data_threads)
data_batches = list(divide_chunks(data_threads, THREADS_NBR))

all_thread_result = pqdm([{"data": db} for db in data_batches], process, n_jobs=THREADS_NBR, argument_type='kwargs')

all_results = []
for thread_result in all_thread_result:
    all_results.extend(thread_result)
print("Total elements processed: ", len(all_results))

print([r["correct_letter"] for r in all_results])
print([r["response"] for r in all_results])

acc = accuracy_score(
    [r["correct_letter"] for r in all_results],
    [r["response"] for r in all_results]
)

recall = recall_score(
    [r["correct_letter"] for r in all_results],
    [r["response"] for r in all_results],
    average = 'weighted'
)

precision = precision_score(
    [r["correct_letter"] for r in all_results],
    [r["response"] for r in all_results],
    average = 'weighted'
)

cm = confusion_matrix(
    [r["correct_letter"] for r in all_results],
    [r["response"] for r in all_results],
)

#TN, FP, FN, TP = cm.ravel()
#specificity = TN / (TN + FP)

print("Accuracy:", acc)
print("Recall:", recall)
#print("Specificity:", specificity)
print("Precision:", precision)

f1_score = classification_report(
    [r["correct_letter"] for r in all_results],
    [r["response"] for r in all_results],
    digits=4,
)

print(f1_score)

with open(f"./results_sft/results_{short_model_name}_{args['variable']}_cot.json", 'w') as f:
    json.dump(all_results, f, indent=4)

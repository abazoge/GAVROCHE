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
parser.add_argument("--variable", type=str, required=True, help="GAVROCHE variable to evaluate")
parser.add_argument("--input_prompt", type=str, required=True, help="Input for evaluation (0-shot or 3-shot)")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 6

base_model_name = args["base_model_name"]
short_base_model_name = base_model_name.split("/")[-1].replace("_","-")

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)

if os.path.isdir('./results_basemodel/') == False:
    os.makedirs('./results_basemodel/', exist_ok=True)

in_corpus = ['facteur_decl', 'premier_episode','adm_arret_cardiaque','atcd_hta','cardiopathie_causale','type_ica','atcd_diabete',
'atcd_insuff_respi_chronique','atcd_bpco','atcd_saos','avc_ait','tabagisme','atcd_tb_rythme','atcd_depression','atcd_cancer','adm_acfa','adm_fevg']

if args["variable"] not in in_corpus:
    print("Error variable name not found!")
    exit(1)

input_prompt = args["input_prompt"]

def process(data):

    results = []

    for current_data in tqdm(data):
        generated_text = current_data['generated'][len(current_data['input_prompt']):].strip()

        results.append({
            "identifier": current_data["identifier"],
            "input_prompt": current_data["input_prompt"],
            "correct_answers": current_data["correct_answers"],
            "generated_text": generated_text
        })

    return results

def divide_chunks(l, n):
    output_chunks = []
    for i in range(0, len(l), n):  
        output_chunks.append(l[i:i + n])
    return output_chunks

dataset = load_from_disk(f"local_hf_{args['variable']}_cot")
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
        data_threads.append({"generated": generated, "input_prompt": d[input_prompt], "identifier": d["identifier"], "correct_answers": d["correct_answers"]})

data_batches = list(divide_chunks(data_threads, THREADS_NBR))

all_thread_result = pqdm([{"data": db} for db in data_batches], process, n_jobs=THREADS_NBR, argument_type='kwargs')

all_results = []
for thread_result in all_thread_result:
    all_results.extend(thread_result)
print("Total elements processed: ", len(all_results))

with open(f"./results_basemodel/results_{short_model_name}_{args['variable']}_cot.json", 'w') as f:
    json.dump(all_results, f, indent=4)

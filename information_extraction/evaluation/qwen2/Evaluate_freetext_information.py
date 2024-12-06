import re
import json
import statistics
from pycm import *
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix

def extract_PA(text):
	text = text.split('**Clinical note:**')[0]
	if '**Answer:**' in text:
		text = text.split('**Answer:**')[1]
	match = re.findall(r'([0-9]+/[0-9]+)', text)
	match2 = re.findall(r'([0-9]+)', text)
	match_total = match+match2
	if len(match) > 0:
		return match
	else:
		return ["Aucune pression artérielle"]

def extract_FC(text):
	text = text.split('**Clinical note:**')[0]
	if '**Answer:**' in text:
		text = text.split('**Answer:**')[1]
	match = re.findall(r'\((\d+)\)', text)
	if len(match) > 0:
		return match
	else:
		return ["Aucune fréquence cardiaque"]

def extract_POIDS(text):
	text = text.split('**Clinical note:**')[0]
	if '**Answer:**' in text:
		text = text.split('**Answer:**')[1]
	match = re.findall(r'\(([0-9]+,[0-9]+)\)', text)
	match2 = re.findall(r'\(([0-9]+\.[0-9]+)\)', text)
	match3 = re.findall(r'\((?<![\.,])(\d+)\)', text)
	match_total = match+match2+match3
	if len(match_total)>0:
		return match_total
	else:
		return ["Aucun poids"]

def extract_TAILLE(text):
	text = text.split('**Clinical note:**')[0]
	if '**Answer:**' in text:
		text = text.split('**Answer:**')[1]
	match = re.findall(r'\(([0-9]+,[0-9]+)\)', text)
	match2 = re.findall(r'\(([0-9]+\.[0-9]+)\)', text)
	match3 = re.findall(r'\(([0-9]Ġ*mĠ*[0-9]+)\)', text)
	match4 = re.findall(r'\((?<![\.,])\)\d+', text)
	match_total = match+match2+match3+match4
	if len(match_total)>0:
		return match_total
	else:
		return ["Aucune taille"]

def extract_IMC(text):
	text = text.split('**Clinical note:**')[0]
	if '**Answer:**' in text:
		text = text.split('**Answer:**')[1]
	match = re.findall(r'\(([0-9]+,[0-9]+)\)', text)
	match2 = re.findall(r'\(([0-9]+\.[0-9]+)\)', text)
	match3 = re.findall(r'\(([0-9]+)\(', text)
	match_total = match+match2+match3
	if len(match_total)>0:
		return match_total
	else:
		return ["Aucun IMC"]

def extract_text_var(text, variable):

	if variable == "adm_clin_pa":
		return extract_PA(text)
	elif variable == "adm_clin_fc":
		return extract_FC(text)
	elif variable == "adm_clin_poids":
		return extract_POIDS(text)
	elif variable == "adm_clin_taille":
		return extract_TAILLE(text)
	elif variable == "adm_clin_imc":
		return extract_IMC(text)
	else: 
		return None


results = []

variables = ['adm_clin_pa', 'adm_clin_fc', 'adm_clin_poids', 'adm_clin_taille', 'adm_clin_imc']

for v in variables:

	print(v)

	# f1 = open(f"./results_fs_cot/results_qwen-2-GAVROCHE-{v.replace('_','-')}-COT-models_{v}_cot.json")
	f1 = open(f"./results_fs_cot/results_Qwen-Qwen2-7B_{v}_cot.json")
	cls_outputs1 = json.load(f1)

	y_labels = []
	y_preds = []
	
	for doc in cls_outputs1:
	
		list_labels = doc['correct_answers']
		list_predictions = extract_text_var(doc['generated_text'], v)
		list_predictions = [l for l in list_predictions if l != None]
	
		for p in list_predictions:
			if p in list_labels:
				y_labels.append(1)
				y_preds.append(1)
				list_labels.remove(p)
			else:
				y_labels.append(0)
				y_preds.append(1)
	
		for l in list_labels:
			y_labels.append(1)
			y_preds.append(0)
	
		if list_labels == [] and list_predictions == []:
			y_labels.append(0)
			y_preds.append(0)

	cm = classification_report(y_labels, y_preds, output_dict=True)
	results.append({
		'variable': v,
		'nb_pos': sum(y_labels),
		'precision': round(cm['1']['precision'], 4),
		'recall': round(cm['1']['recall'], 4),
		'f1_score': round(cm['1']['f1-score'], 4),
	})

with open(f"./results_textvar_llms_FS_COT.json", 'w') as f:
	json.dump(results, f, indent=4)

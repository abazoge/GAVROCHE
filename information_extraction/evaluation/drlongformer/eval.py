import os
import json
import jsonlines
from pycm import *
import statistics
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix

def getConfig(raw_labels):

	label2id = {}
	id2label = {}

	for i, class_name in enumerate(raw_labels):
		label2id[class_name] = str(i)
		id2label[str(i)] = class_name

	return label2id, id2label

def func_final_labels(labels):
	liste_diabete = ['PAS_DE_DIABETE', 'ATCD_DIABETE_TYPE1', 'ATCD_DIABETE_TYPE2', 'ATCD_DIABETE_AUTRE']
	liste_fevg = ['ADM_FEVG_40', 'ADM_FEVG_41_49', 'ADM_FEVG_50']
	liste_tabagisme = ['TABAC_JAMAIS', 'TABAC_SEVRE', 'TABAC_ACTIF']
	liste_fact_decl = ['FD_TB_DU_RYTHME', 'FD_ISCHEMIQUE', 'FD_POUSSEE_HTA', 'FD_INFECTIEUSE', 'FD_REGIME_TRAITEMENT']
	list_cardiopathie_causale = ['CC_ISCH', 'CC_VALV', 'CC_RYTHM']

	if 'CC_AUTRE' in labels or "CC_NON_CONNUE" in labels:
		labels = [j for k,j in enumerate(labels) if j != 'CC_AUTRE' and j != "CC_NON_CONNUE"]
		labels.append('CC_AUTRE_INCONNU')
	elif len(list(set([j for j in labels if j in list_cardiopathie_causale]))) == 0:
		labels.append('CC_AUTRE_INCONNU')

	if len(list(set([j for j in labels if j in liste_fact_decl]))) == 0:
		labels.append('FD_AUTRE_INCONNU')
	
	if 'PREMIER_EPISODE_ICA_OUI' in labels and "PREMIER_EPISODE_ICA_NON" in labels:
		labels = [j for k,j in enumerate(labels) if j != 'PREMIER_EPISODE_ICA_OUI']
	elif 'PREMIER_EPISODE_ICA_OUI' not in labels and "PREMIER_EPISODE_ICA_NON" not in labels:
		labels.append("PREMIER_EPISODE_ICA_NON")
	
	if len(list(set([j for j in labels if j in liste_diabete]))) > 1:
		labels = [j for k,j in enumerate(labels) if j not in liste_diabete]
		labels.append('ATCD_DIABETE_AUTRE')
		
	if len(list(set([j for j in labels if j in liste_fevg]))) > 1:
		fevg_doc = list(set([j for j in labels if j in liste_fevg]))
		if 'ADM_FEVG_40' in fevg_doc:
			labels = [j for k,j in enumerate(labels) if j != 'ADM_FEVG_41_49' and j != 'ADM_FEVG_50']
		elif 'ADM_FEVG_41_49' in fevg_doc:
			labels = [j for k,j in enumerate(labels) if j != 'ADM_FEVG_50']
			
	if len(list(set([j for j in labels if j in liste_tabagisme]))) > 1:
		tabagisme_doc = list(set([j for j in labels if j in liste_tabagisme]))
		if 'TABAC_SEVRE' in tabagisme_doc:
			labels = [j for k,j in enumerate(labels) if j != 'TABAC_JAMAIS' and j != 'TABAC_ACTIF']
		elif 'TABAC_ACTIF' in tabagisme_doc:
			labels = [j for k,j in enumerate(labels) if j != 'TABAC_JAMAIS']
		
	return labels

liste_taille_train = ['50', '100', '150', '200', '250', '300']

blocs = ['1', '2', '3', '4', '5']

runs = ['0', '1', '2', '3', '4']

dict_variables = {
	'1' : ['O', 'B-FD_TB_DU_RYTHME', 'I-FD_TB_DU_RYTHME', 'B-FD_ISCHEMIQUE', 'I-FD_ISCHEMIQUE', 
							'B-FD_POUSSEE_HTA', 'I-FD_POUSSEE_HTA', 'B-FD_INFECTIEUSE', 'I-FD_INFECTIEUSE', 'B-FD_REGIME_TRAITEMENT', 'I-FD_REGIME_TRAITEMENT',
							'B-FD_AUTRE_INCONNU', 'I-FD_AUTRE_INCONNU', 'B-PREMIER_EPISODE_ICA_OUI', 'I-PREMIER_EPISODE_ICA_OUI', 'B-PREMIER_EPISODE_ICA_NON',
							'I-PREMIER_EPISODE_ICA_NON', 'B-TTT_HABITUEL', 'I-TTT_HABITUEL', 'B-ADM_ARRET_CARDIAQUE', 'I-ADM_ARRET_CARDIAQUE'],
	'2' : ['O', 'B-ATCD_HTA',
							'I-ATCD_HTA', 'B-CC_ISCH', 'I-CC_ISCH', 'B-CC_VALV', 'I-CC_VALV', 'B-CC_RYTHM', 'I-CC_RYTHM', 'B-CC_AUTRE', 'I-CC_AUTRE', 'B-CC_NON_CONNUE', 'I-CC_NON_CONNUE', 'B-ICA_TYPE_ICD_ISOLEE', 'I-ICA_TYPE_ICD_ISOLEE',
							'B-ICA_TYPE_ICC_DECOMP', 'I-ICA_TYPE_ICC_DECOMP', 'B-ICA_TYPE_OAP', 'I-ICA_TYPE_OAP', 'B-ICA_TYPE_CHOC_CARDIO', 'I-ICA_TYPE_CHOC_CARDIO'],
	'3' : ['O', 'B-ATCD_INSUFF_RESPI_CHRONIQUE', 'I-ATCD_INSUFF_RESPI_CHRONIQUE', 'B-ATCD_BPCO', 'I-ATCD_BPCO', 'B-ATCD_SAOS', 'I-ATCD_SAOS', 'B-AVC_AIT',
							'I-AVC_AIT', 'B-PAS_DE_DIABETE', 'I-PAS_DE_DIABETE', 'B-ATCD_DIABETE_TYPE1', 'I-ATCD_DIABETE_TYPE1', 'B-ATCD_DIABETE_TYPE2',
							'I-ATCD_DIABETE_TYPE2', 'B-ATCD_DIABETE_AUTRE', 'I-ATCD_DIABETE_AUTRE'],
	'4' : ['O', 'B-ADM_CLIN_FC', 'I-ADM_CLIN_FC', 'B-ADM_CLIN_PA', 'I-ADM_CLIN_PA',
							'B-ADM_CLIN_POIDS', 'I-ADM_CLIN_POIDS', 'B-ADM_CLIN_TAILLE', 'I-ADM_CLIN_TAILLE', 'B-ADM_CLIN_IMC', 'I-ADM_CLIN_IMC',
							'B-TABAC_JAMAIS', 'I-TABAC_JAMAIS', 'B-TABAC_SEVRE', 'I-TABAC_SEVRE', 'B-TABAC_ACTIF', 'I-TABAC_ACTIF'],
	'5' : ['O', 'B-ATCD_TB_RYTHME', 'I-ATCD_TB_RYTHME',
							'B-ATCD_DEPRESSION', 'I-ATCD_DEPRESSION', 'B-ATCD_TB_COGNITIFS', 'I-ATCD_TB_COGNITIFS', 'B-ATCD_CANCER', 'I-ATCD_CANCER',
							'B-ADM_ACFA', 'I-ADM_ACFA', 'B-ADM_FEVG_40', 'I-ADM_FEVG_40', 'B-ADM_FEVG_41_49', 'I-ADM_FEVG_41_49', 'B-ADM_FEVG_50', 'I-ADM_FEVG_50']
}


for ltt in liste_taille_train:

	for b in blocs:

		for r in runs:

			with open(f'./outputs_{ltt}cr/results_DrLongformer_bloc{b}_{ltt}cr_run{r}.json', "r") as data_f:
				json_obj = json.load(data_f)

				names_bloc = dict_variables[b]

				names_bloc_b = [i for i in names_bloc if i[0] == 'O' or i[0] == 'B']

				label2id, id2label = getConfig(names_bloc)

				labels_bloc = [i[2:] for i in names_bloc_b if i[0]=='B']

				final_list_doc = []

				for i in json_obj:
					
					labels_b = [j[2:] for j in i['true_labels'] if j[0] == 'B']
					labels_b = func_final_labels(labels_b)
					final_labels = list(set(labels_b))
					
					predictions = [j[2:] for j in i['true_predictions'] if j[0] == 'B']
					predictions = func_final_labels(predictions)
					final_predictions = list(set(predictions))
					
					i['final_labels'] = final_labels
					i['final_predictions'] = final_predictions

					final_list_doc.append(i)

				oh_variable = {}

				for i in labels_bloc:
					if i != 'ADM_ARRET_CARDIAQUE' and i != 'TABAC_JAMAIS':
						oh_labels = [1 if i in j['final_labels'] else 0 for j in final_list_doc]
						oh_predictions = [1 if i in j['final_predictions'] else 0 for j in final_list_doc]
						oh_variable[i] = {'labels': oh_labels, 'predictions': oh_predictions}

				results = []

				for key, value in oh_variable.items():
					cm = classification_report(value['labels'], value['predictions'], output_dict=True)
					if sum(value['labels']) != 0:
						results.append({
							'variable': key,
							'nb_pos': sum(value['labels']),
							'precision': round(cm['1']['precision'], 4),
							'recall': round(cm['1']['recall'], 4),
							'f1_score': round(cm['1']['f1-score'], 4),
						})
					else:
						results.append({
							'variable': key,
							'nb_pos': sum(value['labels']),
							'precision': 0,
							'recall': 0,
							'f1_score': 0,
						})

				if not os.path.exists(f"./results_{ltt}cr"):
					os.makedirs(f"results_{ltt}cr")
				with open(f"./results_{ltt}cr/results_bloc{b}_run{r}.json", 'w') as f:
					json.dump(results, f, indent=4)

		with open(f'./results_{ltt}cr/results_bloc{b}_run0.json', "r") as d:
			data_run0 = json.load(d)

		with open(f'./results_{ltt}cr/results_bloc{b}_run1.json', "r") as d:
			data_run1 = json.load(d)

		with open(f'./results_{ltt}cr/results_bloc{b}_run2.json', "r") as d:
			data_run2 = json.load(d)

		with open(f'./results_{ltt}cr/results_bloc{b}_run3.json', "r") as d:
			data_run3 = json.load(d)

		with open(f'./results_{ltt}cr/results_bloc{b}_run4.json', "r") as d:
			data_run4 = json.load(d)

		results_mean = []

		for i in range(len(data_run0)):
			precision = statistics.mean([data_run0[i]['precision'], data_run1[i]['precision'], data_run2[i]['precision'], data_run3[i]['precision'], data_run4[i]['precision']])
			recall = statistics.mean([data_run0[i]['recall'], data_run1[i]['recall'], data_run2[i]['recall'], data_run3[i]['recall'], data_run4[i]['recall']])
			f1_score = statistics.mean([data_run0[i]['f1_score'], data_run1[i]['f1_score'], data_run2[i]['f1_score'], data_run3[i]['f1_score'], data_run4[i]['f1_score']])
			
			results_mean.append({
				'variable': data_run0[i]['variable'],
				'nb_pos': data_run0[i]['nb_pos'],
				'precision': round(precision, 4),
				'recall': round(recall, 4),
				'f1_score': round(f1_score, 4),
			})
		if not os.path.exists(f"./results_mean_{ltt}cr"):
			os.makedirs(f"results_mean_{ltt}cr")
		with open(f"./results_mean_{ltt}cr/results_{ltt}cr_bloc{b}.json", 'w') as f:
			json.dump(results_mean, f, indent=4)








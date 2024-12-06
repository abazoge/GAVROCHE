
##### Evaluate SFT model

### Run eval variables
declare -a VARIABLES=("adm_acfa" "adm_arret_cardiaque" "atcd_bpco" "atcd_cancer" "atcd_depression" "atcd_insuff_respi_chronique" "atcd_saos" "atcd_tb_rythme" "avc_ait" "premier_episode" "adm_fevg" "atcd_diabete" "tabagisme")

for variable in ${VARIABLES[@]}; do
        eval "python ./eval_SFT/EvaluateSFT_fewshot_cot.py --peft_name='qwen-2_GAVROCHE_$variable_COT-models' --base_model_name='qwen-2_GAVROCHE_atcd_hta_COT-models' --variable='$variable' --input_prompt='prompt_no_answer_3shot'")
done

### Run eval multilabel variables 
declare -a VARIABLES=("cardiopathie_causale" "facteur_decl" "type_ica")

for variable in ${VARIABLES[@]}; do
        eval "python ./eval_SFT/EvaluateSFT_fewshot_cot_multi.py --peft_name='qwen-2_GAVROCHE_$variable_COT-models' --base_model_name='qwen-2_GAVROCHE_atcd_hta_COT-models' --variable='$variable' --input_prompt='prompt_no_answer_3shot'")
done

### Run eval freetext variables
declare -a VARIABLES=("adm_clin_pa" "adm_clin_fc" "adm_clin_poids" "adm_clin_taille" "adm_clin_imc")

for variable in ${VARIABLES[@]}; do
        eval "python ./eval_SFT/EvaluateSFT_fewshot_cot_freetext.py --peft_name='qwen-2_GAVROCHE_$variable_COT-models' --base_model_name='qwen-2_GAVROCHE_atcd_hta_COT-models' --variable='$variable' --input_prompt='prompt_no_answer_3shot'")
done

python Evaluate_freetext_information.py

##### Evaluate base model

### Run eval variables
declare -a VARIABLES=("adm_acfa" "adm_arret_cardiaque" "atcd_bpco" "atcd_cancer" "atcd_depression" "atcd_insuff_respi_chronique" "atcd_saos" "atcd_tb_rythme" "avc_ait" "premier_episode" "adm_fevg" "atcd_diabete" "tabagisme")

for variable in ${VARIABLES[@]}; do
        eval "python ./eval_base_model/EvaluateSFT_fewshot_cot.py --peft_name='qwen-2_GAVROCHE_$variable_COT-models' --base_model_name='qwen-2_GAVROCHE_atcd_hta_COT-models' --variable='$variable' --input_prompt='prompt_no_answer_3shot'")
done

### Run eval multilabel variables 
declare -a VARIABLES=("cardiopathie_causale" "facteur_decl" "type_ica")

for variable in ${VARIABLES[@]}; do
        eval "python ./eval_base_model/EvaluateSFT_fewshot_cot_multi.py --peft_name='qwen-2_GAVROCHE_$variable_COT-models' --base_model_name='qwen-2_GAVROCHE_atcd_hta_COT-models' --variable='$variable' --input_prompt='prompt_no_answer_3shot'")
done

### Run eval freetext variables
declare -a VARIABLES=("adm_clin_pa" "adm_clin_fc" "adm_clin_poids" "adm_clin_taille" "adm_clin_imc")

for variable in ${VARIABLES[@]}; do
        eval "python ./eval_base_model/EvaluateSFT_fewshot_cot_freetext.py --peft_name='qwen-2_GAVROCHE_$variable_COT-models' --base_model_name='qwen-2_GAVROCHE_atcd_hta_COT-models' --variable='$variable' --input_prompt='prompt_no_answer_3shot'")
done

python Evaluate_freetext_information.py
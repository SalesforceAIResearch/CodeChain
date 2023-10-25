split=test # test OR mini_val
model=gpt3.5 # gpt3.5 or gpt4 
start=0
end=10 
num_gen_samples=20
start_round=1
end_round=5

## ROUND 0 
prompt=prompts/codechain_gen.txt 
round=round0
exp_name=${model}_${split}
output_path=outputs/${exp_name}_$round
num_clusters=5

# Generate code 
python src/generate.py --output_path $output_path --prompt_file $prompt --split $split --model $model --start $start --end $end --num_gen_samples $num_gen_samples 

# Test by example test cases 
python src/evaluate.py --save_gen_path $output_path --example_test_path data/${split}_example_tests.pkl  --eval_split $split --save_results_path ${output_path}_results_exampletests.json 

# Postprocessing 
python src/processing.py --output_path $output_path --output_file ${output_path}_data.csv --result_path ${output_path}_results_exampletests.json 

# Clustering 
echo "Clustering N_CLUSTERS: $num_clusters" 
python src/clustering.py --data_path ${output_path}_data.csv --n_cluster ${num_clusters} --output_embed_file ${output_path}_embeds.npy --output_file ${output_path}_${num_clusters}clusters.csv


## REVISION ROUNDS 
prompt=prompts/codechain_revise.txt  

for (( round_number = $start_round; round_number <= $end_round; round_number++ )) 
do
    echo "REVISION ROUND $round_number"

    prior_output_path=$output_path
    prior_num_clusters=$num_clusters 
    prior_result_path=${prior_output_path}_data.csv_all_results.json
    modules_file=${prior_output_path}_${prior_num_clusters}clusters.csv
    echo "MODULE FILE $modules_file"

    round=round${round_number}
    output_path=outputs/${exp_name}_$round
    num_clusters=$(( 5 - $round_number ))
    echo "OUTPUT PATH: $output_path"

    # generate code 
    python src/generate.py --output_path $output_path --prompt_file $prompt --split $split --model $model --start $start --end $end --num_gen_samples $num_gen_samples --modules_file $modules_file --num_clusters ${prior_num_clusters}

    if [ $round_number -ne $end_round ]
    then
        # Test by example test cases 
        python src/evaluate.py --save_gen_path $output_path --example_test_path data/${split}_example_tests.pkl  --eval_split $split --save_results_path ${output_path}_results_exampletests.json 

        # Postprocessing 
        python src/processing.py --output_path $output_path --output_file ${output_path}_data.csv --result_path ${output_path}_results_exampletests.json --past_result_path $prior_result_path

        # Clustering 
        echo "Clustering N_CLUSTERS: $num_clusters" 
        python src/clustering.py --data_path ${output_path}_data.csv --n_cluster ${num_clusters} --output_embed_file ${output_path}_embeds.npy --output_file ${output_path}_${num_clusters}clusters.csv
    fi 
done

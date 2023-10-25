split=test # test OR mini_val
model=gpt3.5 # gpt3.5 or gpt4 
start=0
end=10
num_gen_samples=20

prompt=prompts/direct_gen.txt 
exp_name=${model}_${split}
output_path=outputs/${exp_name}_directgen 

python src/evaluate.py --save_gen_path $output_path --eval_split $split
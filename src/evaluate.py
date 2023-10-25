import tempfile, shutil, os
import json
import pprint 
import pickle as pkl 

from datetime import timedelta
import time
from copy import copy
import random
from copy import deepcopy
from functools import partial
import shutil
from glob import glob 
from tqdm import tqdm 
from scipy.stats import describe 
import random 
import pdb 

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from configs.config_evaluate import parse_args
from utils.utils import extract_code  
from utils.utils_evaluate import safe_eval_answer_from_agent, compute_pass_at_ks

args = parse_args()
argsdict = vars(args)
print(pprint.pformat(argsdict))

val_dataset = load_from_disk(f'data/{args.eval_split}')
    
if args.example_test_path is not None: 
    example_tests = pkl.load(open(args.example_test_path, 'rb'))

num_outputs = [] 
original_samples = 0
processed_val_dataset = DatasetDict() # empty DatasetDict 
for level, subset_data in val_dataset.items():
    new_data = [] 
    for sample in subset_data: 
        problem_id = sample['problem_id'] 

        starter_code = sample['starter_code']
        gen_file = args.save_gen_path + '/{}.json'.format(problem_id)

        if not os.path.exists(gen_file) and args.original_gen_path is not None:
            gen_file = args.original_gen_path + '/{}.json'.format(problem_id)
            original_samples += 1 

        if os.path.exists(gen_file):
            data = json.load(open(gen_file, 'r'))
            output = data['output']
            prompt = data['prompt']
            final_code = [extract_code(c) for c in output]
           
        else:
            continue 
            
        if len(final_code)==0: 
            final_code = ['']

        num_outputs.append(len(final_code))
        sample['gpt_codes'] = final_code 

        sample['gen_file'] = gen_file 

        if args.example_test_path is not None: 
            tests = example_tests[sample['problem_id']]
            if tests is None: 
                continue 
            else:
                sample['input_output'] = tests 

        new_data.append(sample)


    processed_val_dataset[level] = Dataset.from_list(new_data)


print("Evaluation dataset:") 
print(processed_val_dataset)
print("# Samples from original (Round 0): {}".format(original_samples))

# generate answer for each difficulty
res_strs = []
res_strs_by_passk = {} 
pass_results_all = {} 
for level, new_subset_data in processed_val_dataset.items():
    new_subset_data = new_subset_data.map(safe_eval_answer_from_agent, num_proc=args.num_proc)
    processed_val_dataset[level] = Dataset.from_list(new_subset_data)
    
    # calculate metrics
    pass_results = [example['gpt_pass_flags'] for example in processed_val_dataset[level] if example['gpt_pass_flags']]
   
    for example in processed_val_dataset[level]:
        pass_results_all[example['problem_id']] = {'result': example['gpt_pass_flags'], 'file': example['gen_file']}
    
    pass_at_ks = compute_pass_at_ks(pass_results, ks=[1]) 
   
    cost = sum([dp['gpt_cost'] if 'gpt_cost' in dp else 0 for dp in processed_val_dataset[level]])
    res_str = f"Difficulty level: {level}, pass@k: {pass_at_ks}, num of examples: {len(pass_results)}, total_cost: {cost}"
    print(res_str)
    res_strs.append(res_str)
    for k,v in pass_at_ks.items(): 
        if k not in res_strs_by_passk:
            res_strs_by_passk[k] = []
        res_strs_by_passk[k].append(v)
print("="*20)
print('\n'.join(res_strs))

print("Number of outputs distributions:")
print(describe(num_outputs))

for k,v in res_strs_by_passk.items():
    v = [str(i) for i in v]
    print('pass@{}'.format(k), ' '.join(v))

if args.save_results_path is not None: 
    json.dump(pass_results_all, open(args.save_results_path, 'w'))
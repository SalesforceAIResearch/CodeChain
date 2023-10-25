import json 
from glob import glob 
from tqdm import tqdm 
from multiprocessing import Pool
from functools import partial
import numpy as np
import re
import argparse, pprint 
import datasets
import os 
from collections import Counter
import pandas as pd

import tokenize
from io import BytesIO
from collections import deque
from scipy.stats import describe 
import itertools

from configs.config_processing import parse_args
from utils.utils import extract_func, get_func_codes, extract_code, is_in_final_code, udpate_code_by_all_past_results

import pdb 

args = parse_args()
argsdict = vars(args)
print(pprint.pformat(argsdict))

files = sorted(glob(args.output_path + '/*.json'))
print("Number of data files: {}".format(len(files)))
    
if args.result_path is not None: 
    results = json.load(open(args.result_path, 'r'))    
    if args.past_result_path is not None: 
        past_results = json.load(open(args.past_result_path, 'r'))
        results, files = udpate_code_by_all_past_results(results, past_results, files)
        
all_funcs = [] 
all_func_list = [] 
all_files = [] 
all_func_codes = [] 

empty_reasoning_count = 0 
empty_funcs_count = 0 
empty_code_count = 0
empty_final_funcs_count = 0

num_outputs = [] 

it = tqdm(files, total=len(files))
for sample in it:
    data = json.load(open(sample, 'r'))
    if 'output' not in data: pdb.set_trace()
    output = data['output']
    problem_id = sample.split('/')[-1].replace('.json','')
    
    if args.result_path is not None:
        if str(problem_id) not in results:
            continue
            
        curr_results = results[str(problem_id)]['result']
        if len(curr_results) != len(output): pdb.set_trace()
        output = [o for i, o in enumerate(output) if curr_results[i]==True]
    
    num_outputs.append(len(output))
    
    final_code = [extract_code(o) for o in output] 
    final_code = [o for o in final_code if len(o)>0]
    
    funcs = [extract_func(o) for o in output]
    func_codes = [get_func_codes(f) for f in final_code]

    assert len(func_codes) == len(funcs)

    temp = [is_in_final_code(f, fc) for f, fc in zip(funcs, func_codes)]
    funcs  = [f[0] for f in temp]
    funcs = list(itertools.chain.from_iterable(funcs))
    full_funcs = [f[1] for f in temp]
    full_funcs = list(itertools.chain.from_iterable(full_funcs))

    # Add functions from final code too 
    if len(funcs)==0 and len(output)!=0:         
        full_funcs = list(itertools.chain.from_iterable(func_codes))
        funcs = ['No docstring'] * len(full_funcs)
        if len(funcs)==0:
            empty_final_funcs_count += 1 
    
    all_funcs += funcs 
    all_func_codes += full_funcs 
    all_func_list.append(funcs)
    
    all_files += [sample] * len(funcs)
    
assert len(all_funcs) == len(all_func_codes)

    
print("Number of generation samples: ")
print(describe(num_outputs))
print(Counter(num_outputs))
print()
functions_per_problem = [len(i) for i in all_func_list]
print("Average func per problem: {}".format(sum(functions_per_problem)/len(functions_per_problem)))
print(describe(functions_per_problem))
print(Counter(functions_per_problem))

if args.output_file is not None: 
    json.dump(results, open(args.output_file + '_all_results.json', 'w'))
    
    processed_data = pd.DataFrame(
        {'file': all_files,
         'func': all_funcs,
         'func_code': all_func_codes
        })

    print("Saving processed data to {}".format(args.output_file))
    processed_data.to_csv(args.output_file)
    

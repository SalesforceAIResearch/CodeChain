import json 
from glob import glob 
import pdb 
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
import random 
import tokenize
from io import BytesIO
from collections import deque
from scipy.stats import describe 
import copy 
import torch 

def extract_code_segment(result, keyword, all_segments=True):
    regex = '\`\`\`\s*{}((?:.|\n)*?)\`\`\`'.format(keyword)
    codes = re.findall(regex, result)
    if len(codes)==0: 
        regex = '\`\`\`\s*{}'.format(keyword)
        indices = [(m.start(0), m.end(0)) for m in re.finditer(regex, result)]
        if len(indices) == 0:
            return ''
        last_end_index = indices[-1][1]
        code = result[last_end_index:]
        return code 
    if all_segments:
        return '\n'.join(codes)
    return codes[-1]

def extract_code(result, keyword='python'):
    code = extract_code_segment(result, keyword)
    if len(code)==0: 
        code = extract_code_segment(result, '', False)

    return code 

def extract_func(result, keyword='module'): 
    regex = '\`\`\`\s*{}((?:.|\n)*?)\`\`\`'.format(keyword)
    codes = re.findall(regex, result)
    
    if len(codes)==0: 
        if '\nSTEP 2' in result:
            index = result.index('\nSTEP 2')
            regex = '\`\`\`\s*python((?:.|\n)*?)\`\`\`'
            codes = re.findall(regex, result[:index])
    
    codes = [o for o in codes if 'class ' not in o and 'def main(' not in o]
        
    new_outputs = [] 
    for output in codes: 
        indices = [m.start() for m in re.finditer('def ', output)]
        if len(indices)>1: 
            funcs = [] 
            for i, index in enumerate(indices[:-1]):
                func = output[index: indices[i+1]]
                funcs.append(func)
            func = output[indices[-1]:]
            funcs.append(func)
            new_outputs += funcs 
        elif len(indices)==0: 
            continue 
        else:
            new_outputs.append(output)
    
    return new_outputs 

def get_func_codes(code_string):
    code_string = code_string.strip()
    file = BytesIO(code_string.encode())
    tokens = None 
    try:
        tokens = deque(tokenize.tokenize(file.readline))
    except Exception as e: 
        print("Error parsing function code: " + str(e)) 
        pass 
    if tokens is None: 
        return []
    lines = []
    while tokens:
        token = tokens.popleft()
        if token.type == tokenize.NAME and token.string == 'def':
            start_line, _ = token.start
            last_token = token
            while tokens:
                token = tokens.popleft()
                if token.type == tokenize.NEWLINE:
                    break
                last_token = token
            if last_token.type == tokenize.OP and last_token.string == ':':
                indents = 0
                while tokens:
                    token = tokens.popleft()
                    if token.type == tokenize.NL:
                        continue
                    if token.type == tokenize.INDENT:
                        indents += 1
                    elif token.type == tokenize.DEDENT:
                        indents -= 1
                        if not indents:
                            break
                    else:
                        last_token = token
            lines.append((start_line, last_token.end[0]))
    code_lines = code_string.split('\n')
    outputs = [] 
    for line in lines: 
        start, end = line 
        function = '\n'.join(code_lines[start-1:end])
        if len(function.strip())>0:
            outputs.append(function)
    return outputs 

def is_in_final_code(funcs, func_codes): 
    output_funcs = [] 
    output_funcs_codes = [] 
    for func in funcs: 
        lines = func.split('\n')
        for line in lines: 
            if 'def ' in line: 
                for func_code in func_codes:
                    if line.strip() in func_code:
                        output_funcs.append(func)
                        output_funcs_codes.append(func_code)
                        break 
                break 
    assert len(output_funcs) == len(output_funcs_codes)
    return output_funcs, output_funcs_codes 

def get_embedding(output_embed_file, data, embedding_model, func_type='func'): 
    from embedding.encoder import CodeBERT, StarEncoder, CodeT5Plus  
    
    if func_type == 'func':
        seqs = [] 
        for x, y in zip(data['func'].tolist(), data['func_code'].tolist()):
            if x != 'No docstring':
                seqs.append(x + '\n' + y)
            else:
                seqs.append(y)
                
    elif func_type == 'centroid':
        seqs = []
        clusters = [] 
        docs = [] 
        codes = [] 
        for row in data.iterrows(): 
            cluster = row[1]['cluster']
            if cluster not in clusters:
                clusters.append(cluster)
                doc = row[1]['centroid']
                code = row[1]['centroid_code']
                docs.append(doc)
                codes.append(code)
                if doc != 'No docstring':
                    seqs.append(doc + '\n' + code)
                else:
                    seqs.append(y)
                
    if os.path.exists(output_embed_file): 
        print("Loading embedding from {}".format(output_embed_file))
        embeds = np.load(output_embed_file)
        print("Embedding of shape {}".format(embeds.shape))
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        MAX_INPUT_LEN = 10000
        MAX_TOKEN_LEN = 512 if embedding_model == 'codebert' else 1024 

        if embedding_model == 'codebert':
            encoder = CodeBERT(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)
        elif embedding_model == 'starencoder': 
            encoder = StarEncoder(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)
        elif embedding_model == 'codet5': 
            encoder = CodeT5Plus(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)
        
        print("Obtain embeddings...")
        embeddings = encoder.encode(seqs)
        embeds = np.stack(embeddings, axis=0)
        print("Embedding of shape {}".format(embeds.shape))
        np.save(output_embed_file, embeds)
        print("Saved embedding to {}".format(output_embed_file))
    
    if func_type == 'centroid':
        return embeds, docs, codes 
    
    return embeds 

def create_func_prompt(doc, code):
    if doc == 'No docstring': 
        return code 
    else:
        code_lines = code.split('\n')
        cutoff_line_idx = -1 
        for line_idx, line in enumerate(code_lines): 
            if 'def ' in line: 
                cutoff_line_idx = line_idx
                break 
        code = '\n'.join(code_lines[cutoff_line_idx+1:])
        return doc + '\n' + code

def get_util_functions_self_cluster(data, num_clusters=1, problem_id_type=int):
    outputs = {}     
    for row in data.iterrows():
        file = row[1]['file']
        problem_id = problem_id_type(file.split('/')[-1].replace('.json', ''))
        centroid = row[1]['centroid_code']
        centroid_doc = row[1]['centroid']

        if problem_id not in outputs: 
            outputs[problem_id] = []
        
        func_str = create_func_prompt(centroid_doc, centroid)
        if func_str not in outputs[problem_id]:
            outputs[problem_id].append(func_str)    
        
    new_outputs = {} 
    for k,v in outputs.items(): 
        sampled = random.sample(v, min(num_clusters, len(v)))
        new_outputs[k] = sampled
    
    lens = [len(i) for i in new_outputs.values()]
    print("Distribution of number of utils:")
    print(describe(lens))
    print(Counter(lens))
    return new_outputs 

def udpate_code_by_all_past_results(results, past_results, files):
    new_files = [] 
    for k,v in past_results.items():
        past_result = v['result']
        
        if k in results: 
            curr_result = results[k]['result']            
        
            # if no passed code in this round, check past results 
            if True not in curr_result and True in past_result: 
                results[k] = past_results[k]
        
        elif True in past_result: 
            results[k] = past_results[k]
            
        if k in results:
            new_files.append(results[k]['file'])        
    
    return results, new_files 
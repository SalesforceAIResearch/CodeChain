import gzip
import io
import itertools
import json
import pprint
import numpy as np
import re
import sys
import timeout_decorator
import traceback
from collections import Counter 
from io import StringIO
import sys
from collections import defaultdict
from datasets import concatenate_datasets, load_dataset
from multiprocessing import Process, Queue
import multiprocessing
from tqdm import tqdm
from typing import Dict, List, Union
import os 
import ast 
import random 
import subprocess
import tempfile, shutil, os
from pyext import RuntimeModule
from copy import deepcopy, copy 
from functools import wraps
import time
import contextlib
import pdb 

from utils.utils_execute import run_test 

GLOBAL_TIMEOUT = 10 # TIMEOUT for one solution

def safe_eval_answer_from_agent(example):
    def _temp_run(code, tests, result):
        try:
            flag, outcomes = verify_code_official(tests, code)
            result.append(flag)
        except Exception as e:
            pass
    
    example['gpt_pass_flags'] = []
    try:
        if type(example['input_output'])==str:
            tests = json.loads(example['input_output'])
        else:
            tests = example['input_output']
    except:
        print(f"Failed to get unit tests for problem {example['problem_id']} with {example['input_output']}")
        return example
    
    for code in example['gpt_codes']:
        manager = multiprocessing.Manager()
        result = manager.list()
        start = time.time()
        p = multiprocessing.Process(target=_temp_run, args=(code, tests, result))
        p.start()
        p.join(timeout=GLOBAL_TIMEOUT + 1)
        time_used = time.time() - start
        if p.is_alive():
            p.kill()
        if not result:
            result = [-1]

        if result[0] == True:
            example['gpt_pass_flags'] += [True]
        else:
            example['gpt_pass_flags'] += [False]

    return example

def verify_code_official(tests, solution, debug=False, return_output=False):
    ''' verify if code passes all tests, using apps official implementation (https://github.com/hendrycks/apps/blob/main/eval/testing_util.py#L122)
    '''
    tests = deepcopy(tests)
    # suppress the stdout of solution execution
    # todo: suppress stderr as well
    with contextlib.redirect_stdout(io.StringIO()):
        results = run_test(tests, solution, debug=debug, return_output=return_output)
        if return_output:
            all_outputs = results[1]
            results = results[0]
    if all([res == True for res in results]):
        if return_output:
            return True, results, all_outputs
        return True, results
    else:
        if return_output:
            return False, results, all_outputs
        return False, results

    
def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    Taken from https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py#L13.
    """
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def compute_pass_at_ks(results, ks):
    output = {
        k: estimate_pass_at_k(
            [len(x) for x in results],
            [sum([i == True for i in x]) for x in results],
            k,
        ).mean()
        for k in ks
    }
    return output

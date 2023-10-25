import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Config for evaluation")
    
    parser.add_argument("--num_proc", default=os.cpu_count(), type=int)
    
    parser.add_argument(
        "--eval_split",
        choices=['mini_val', 'test'],
        default='mini_val',
        help="Which split to evaluate in APPS",
    )
    
    parser.add_argument(
        "--save_gen_path",
        type=str,
        default=None, 
        help='Path to generated code to be evaluated'
    )
    
    parser.add_argument(
        "--save_results_path",
        type=str,
        default=None, 
        help='Path to save test results'
    )
    
    parser.add_argument(
        "--example_test_path",
        type=str,
        default=None, 
        help='Path to example test cases if testing on example test cases; if None, test on hidden test cases'
    )
    
    parser.add_argument(
        "--original_gen_path",
        type=str,
        default=None, 
        help='Path to the original generation code e.g. round 0 generation; if None, only test on revised programs'
    )
        
    args = parser.parse_args()

    return args

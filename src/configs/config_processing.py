import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Config for post-processing generated code to extract code and modular functions")

    parser.add_argument(
        '--output_path',
        type=str,
        default=None, 
        help="Path to generated code to be processed")
    parser.add_argument(
        '--result_path',
        type=str,
        default=None, 
        help="Path to test results on example test cases for filtering")
    parser.add_argument(
        '--past_result_path',
        type=str,
        default=None, 
        help="Path to all past test results (compiled from all prior generation rounds) on example test cases")
    parser.add_argument(
        '--output_file',
        type=str,
        help="Path to save processed data")
    
    args = parser.parse_args()

    return args
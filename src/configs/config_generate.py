import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Config for generation")
    
    parser.add_argument(
        '--model',
        type=str,
        help="Model name for generation")
    parser.add_argument(
        '--output_path',
        type=str,
        help="Output path to save generated code")
    parser.add_argument(
        '--prompt_file',
        type=str,
        help="Path to instruction prompt")
    parser.add_argument(
        '--modules_file',
        type=str,
        default=None, 
        help="Path to extracted modules for self-revision")
    parser.add_argument(
        '--num_gen_samples',
        type=int,
        default=5,
        help="Number of generation samples per problem")
    parser.add_argument(
        '--split',
        type=str,
        default='mini_val',
        help="name of data split in APPS")
    parser.add_argument(
        '--num_clusters',
        type=int,
        default=1,
        help="Number of clusters in prior generation round")
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help="Star index of dataset")
    parser.add_argument(
        '--end',
        type=int,
        default=150,
        help="End index of dataset")
    
    args = parser.parse_args()

    return args

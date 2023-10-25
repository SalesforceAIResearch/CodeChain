import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Config for clustering")
    
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='starencoder', 
        help="Type of embedding model; default to use StarCoder encoder")
    parser.add_argument(
        '--data_path',
        type=str,
        default=None, 
        help="Path to generated code for clustering")
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=5, 
        help="Number of clusters to run k-means clustering")
    parser.add_argument(
        '--output_embed_file',
        type=str,
        default=None, 
        help="Path to save embedding file")
    parser.add_argument(
        '--output_file',
        type=str,
        default=None, 
        help="Path to save processed data of clustered modular functions")
    
    args = parser.parse_args()

    return args
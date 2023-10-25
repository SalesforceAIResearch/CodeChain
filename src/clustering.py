import json 
from glob import glob 
from tqdm import tqdm 
from multiprocessing import Pool
from functools import partial
import numpy as np
import re
import datasets
import pickle as pkl 
import pandas as pd 
import argparse, pprint 
from collections import Counter 
from scipy.stats import describe 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from configs.config_clustering import parse_args
from utils.utils import get_embedding

args = parse_args()
argsdict = vars(args)
print(pprint.pformat(argsdict))

data = pd.read_csv(args.data_path)
embeds = get_embedding(args.output_embed_file, data, args.embedding_model)

# Cluster by embeddings
print("Clustering...") 
groups_by_problem = data.groupby(['file']).groups
funcs = data['func'].tolist()
func_codes = data['func_code'].tolist()
all_files = []
all_centroids = []
all_centroid_codes = [] 
num_funcs = [] 
all_distances = [] 

all_num_clusters = []
for group, indices in tqdm(groups_by_problem.items()):
    num_funcs.append(len(indices))
    group_embeds = embeds[indices]
    
    num_clusters = args.n_clusters 
    all_num_clusters.append(num_clusters)

    kmeans_model = KMeans(n_clusters=min(len(indices), args.n_clusters), random_state=0)
    classes = kmeans_model.fit_predict(group_embeds).tolist()
    
    min_dist = np.min(cdist(group_embeds, kmeans_model.cluster_centers_, 'euclidean'), axis=1)
    
    Y = pd.DataFrame(min_dist, index=range(len(classes)), columns=['Center_euclidean_dist'])
    Z = pd.DataFrame(classes, index=range(len(classes)), columns=['cluster_ID'])

    PAP = pd.concat([Y,Z], axis=1)
    grouped = PAP.groupby(['cluster_ID'])
    centroids = grouped.idxmin()    
        
    centroids_mapping = centroids.to_dict()['Center_euclidean_dist']
    for cluster, index in centroids_mapping.items(): 
        real_index = indices[index]
        all_centroids.append(funcs[real_index])
        all_centroid_codes.append(func_codes[real_index])
        all_files.append(group)

print("Number of functions per problem:")
print(describe(num_funcs))
print() 
data = {} 
data['file'] = all_files
data['centroid'] = all_centroids
data['centroid_code'] = all_centroid_codes
    
print("number of set clusters:")
print(Counter(all_num_clusters))
print(describe(all_num_clusters))

freqs = Counter(all_files)
print("Number of util functions per problem:")
print(describe(list(freqs.values())))

data = pd.DataFrame(data)
if args.output_file is not None: 
    data.to_csv(args.output_file)
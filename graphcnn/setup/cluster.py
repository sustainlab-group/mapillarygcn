import datetime
import json
from itertools import count

import scipy.io
import numpy as np
import pandas as pd
import h5py

from graphcnn.helper import *
import graphcnn.setup.helper
import graphcnn.setup as setup
import math
from math import radians, cos, sin, asin, sqrt 
from scipy import stats

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import random
from pathlib import Path

MAX_DIST = 19.890915

def get_invdistance(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 
    
    dlon = abs(lon2 - lon1)  
    dlat = abs(lat2 - lat1) 
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    c = 2 * asin(sqrt(a))  
     
    # Radius of earth is 6371 kilometers, or 3956 miles 
    r = 6371
    
    return 1 - (float(c * r) / MAX_DIST)

def get_imgfeat_similarity(f1, f2):
    return cosine_similarity(np.array([f1]),np.array([f2]))


def jaccard_similarity(arr1, arr2):
    s1 = set(arr1)
    s2 = set(arr2)
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))


def get_obj_count_similarity(f1, f2, obj_sim='jaccard'):
    # sim can be 'cosine' or 'jaccard'
    if obj_sim == 'cosine':
        if np.sum(f1) == 0 or np.sum(f2) == 0:
            return 0.
        return cosine_similarity(np.array([f1]),np.array([f2]))[0][0]
    else:
        return jaccard_similarity(f1, f2)

def get_single_obj_count_similarity(f1, f2, obj_sim='jaccard', normalize=False):
    # sim can be 'cosine' or 'jaccard'
    obj_sim = []
    diff = np.maximum(f1, f2) - np.absolute(f1 - f2)
    return diff

def process_obj_counts(counts):
    NUM_FEATS = 66 # num_stuff == 28, num_thing == 37, 255 is void class
    try:
        
        counts = [NUM_FEATS - 1 if x == 255 else x for x in counts]
        counts = np.histogram(counts, bins=np.arange(0, NUM_FEATS))[0]
    except:
        counts = np.zeros(NUM_FEATS) 
    return counts

def get_single_obj_counts(counts):
    NUM_FEATS = 66
    single_counts = [0] * NUM_FEATS
    try:
        for x in counts:
            if x == 255:
                single_counts[NUM_FEATS - 1] += 1
            else:
                single_counts[x] += 1
    except:
        pass
    #print(single_counts)
    return single_counts
                


def load_cluster_dataset(target, img_csv, train_val_dir, h5py_fn_raw, 
                         V_type='both', A_type='inv', obj_sim='jaccard', 
                         CLUSTER_LIMIT=-1):
    # Load the images with their cluster_ids, target labels, and coords
    print("Loading images...")
    #img_csv = "/atlas/u/jihyeonlee/segment_mapillary/pipeline/povpopbmi_labels_per_image_224x224_rescale.csv" 
    img_df = pd.read_csv(img_csv)
    cluster_ids = img_df['unique_cluster'].to_numpy()
    targets = img_df[target].to_numpy()
    lats = img_df['ilat'].to_numpy()
    lons = img_df['ilon'].to_numpy()
    obj_counts = img_df['features'].to_numpy()
    obj_counts = list(map(json.loads, obj_counts))
    pad = len(max(obj_counts, key=len))
    obj_counts = np.array([i + [0]*(pad-len(i)) for i in obj_counts])
    #obj_counts = np.vstack(obj_counts[:])
    
    if obj_sim:# == 'cosine':
        obj_fn = 'count_histogram.csv'
        obj_pkl_fp = Path(obj_fn)
        if not obj_pkl_fp.is_file():
        #obj_counts = np.array(list(map(process_obj_counts, obj_counts)))
            print("Collecting object counts from scratch...")
            obj_counts = np.apply_along_axis(get_single_obj_counts, 1, obj_counts)
            print(obj_counts.shape)
            np.savetxt(obj_fn, obj_counts, delimiter=",")
        else:
            print("Loading presaved object counts...")
            obj_counts = np.loadtxt(obj_fn, delimiter=',')
    obj_counts_V = img_df['features'].to_numpy()
    obj_counts_V = np.array(list(map(json.loads, obj_counts_V)))
    obj_counts_V = np.array(list(map(process_obj_counts, obj_counts_V)))
    
    NUM_FEATURES = 66

    # Load the clusters
    print("Loading clusters...")
    #'/atlas/u/jihyeonlee/segment_mapillary/pipeline/poverty/ia/train_clusters_ia.txt'
    #'/atlas/u/jihyeonlee/segment_mapillary/pipeline/poverty/ia/val_clusters_ia.txt'
    train_f = open(train_val_dir + 'train_clusters_ia.txt', 'r')
    val_f = open(train_val_dir + 'val_clusters_ia.txt', 'r')
    train_clusters = [x[:-1] for x in train_f.readlines()]
    val_clusters = [x[:-1] for x in val_f.readlines()]
    train_f.close()
    val_f.close()

    # Load the pretrained image features
    print("Loading pretrained image features...")

    h5py_fn = h5py_fn_raw if len(h5py_fn_raw) > 0 else '/atlas/u/jihyeonlee/segment_mapillary/pipeline/pretrained_on_pov.hdf5' 
    dataset = h5py.File(h5py_fn, 'r')

    img_feats = dataset['features']

    print("Collecting V and A...")
    max_no_imgs = 200
    imagnet_feature_size = 512
    labels, labels_test = [], [] 
    clusters_V, clusters_V_test = [], []
    clusters_A, clusters_A_test = [], []
    
    CNT_LIMIT, cnt = -1, 0 
    PERCENTILE_CUTOFF = 50
    
    # Collect training data
    for cluster_id in tqdm(train_clusters):
        if cnt < int(0.8 * CNT_LIMIT) or CNT_LIMIT == -1:
            imgs_in_cluster = np.argwhere(cluster_ids == cluster_id).flatten() # indexes of relevant rows
            if imgs_in_cluster.shape[0] < CLUSTER_LIMIT: # empty cluster
                continue

            # Get label
            label = targets[imgs_in_cluster[0]]
            labels.append(label)

            # Get nodes
            if V_type == 'feats' or V_type == 'both':
                V = img_feats[imgs_in_cluster, :, :, :].squeeze(axis=-1).squeeze(axis=-1)
                if V.shape[0] < 200: # then pad
                    V = np.pad(V, ((0,200-V.shape[0]), (0,0)))
            if V_type == 'obj':
                V = obj_counts_V[imgs_in_cluster, :]
                if V.shape[0] < 200: # then pad
                    V = np.pad(V, ((0,200-V.shape[0]), (0,0)))
            if V_type == 'both':
                V_obj = obj_counts_V[imgs_in_cluster, :]
                if V_obj.shape[0] < 200: # then pad
                    V_obj = np.pad(V_obj, ((0,200-V_obj.shape[0]), (0,0)))
                norm = np.linalg.norm(V_obj)
                V_obj = V_obj / norm  if norm != 0 else V_obj
                V = np.concatenate((V, V_obj), axis=1)
            clusters_V.append(V)
            
            # Get edges
            A = np.zeros([max_no_imgs, max_no_imgs], dtype=np.float32)
            if obj_sim:
                A_obj_single = np.zeros([max_no_imgs, NUM_FEATURES, max_no_imgs])
            if A_type != 'none':
                for i, row_idx_i in enumerate(imgs_in_cluster):
                    p1 = (float(lats[row_idx_i]), float(lons[row_idx_i]))
                    for j, row_idx_j in enumerate(imgs_in_cluster):
                        if A_type == 'inv' or A_type == 'both':
                            p2 = (float(lats[row_idx_j]), float(lons[row_idx_j]))
                            A[i, j] = get_invdistance(p1, p2) if i != j else float(1)
                        if obj_sim:
                            A_obj_single[i, :, j] = get_single_obj_count_similarity(obj_counts[i], obj_counts[j], obj_sim=obj_sim)

                if obj_sim:
                    A_obj_single /= float(np.amax(A_obj_single))

            elif A_type == 'none':
                A = np.random.rand(max_no_imgs, max_no_imgs)
            else:
                raise
            A = np.expand_dims(A, axis=1)
            if obj_sim:
                A = np.concatenate((A, A_obj_single), axis=1)
        clusters_A.append(A)
        cnt += 1
    
    # Collect test data
    cnt = 0
    for cluster_id in tqdm(val_clusters):
        if cnt < int(0.2 * CNT_LIMIT) or CNT_LIMIT == -1:
            imgs_in_cluster = np.argwhere(cluster_ids == cluster_id).flatten() # indexes of relevant rows
            if imgs_in_cluster.shape[0] < CLUSTER_LIMIT: # empty cluster
                continue
                

            # Get label
            label_test = targets[imgs_in_cluster[0]]
            labels_test.append(label_test)

            # Get nodes
            if V_type == 'feats' or V_type == 'both':
                V = img_feats[imgs_in_cluster, :, :, :].squeeze(axis=-1).squeeze(axis=-1)
                if V.shape[0] < 200: # then pad
                    V = np.pad(V, ((0,200-V.shape[0]), (0,0)))
            if V_type == 'obj':
                V = obj_counts_V[imgs_in_cluster, :]
                if V.shape[0] < 200: # then pad
                    V = np.pad(V, ((0,200-V.shape[0]), (0,0)))
            if V_type == 'both':
                V_obj = obj_counts_V[imgs_in_cluster, :]
                if V_obj.shape[0] < 200: # then pad
                    V_obj = np.pad(V_obj, ((0,200-V_obj.shape[0]), (0,0)))
                norm = np.linalg.norm(V_obj)
                V_obj = V_obj / norm  if norm != 0 else V_obj
                V = np.concatenate((V, V_obj), axis=1)
            clusters_V_test.append(V)

            # Get edges 
            A = np.zeros([max_no_imgs, max_no_imgs], dtype=np.float32)
            if obj_sim:
                A_obj = np.zeros([max_no_imgs, max_no_imgs], dtype=np.float32)
                A_obj_single = np.zeros([max_no_imgs, NUM_FEATURES, max_no_imgs])
            if A_type != 'none':
                for i, row_idx_i in enumerate(imgs_in_cluster):
                    p1 = (float(lats[row_idx_i]), float(lons[row_idx_i]))
                    for j, row_idx_j in enumerate(imgs_in_cluster):
                        if A_type == 'inv' or A_type == 'both':
                            p2 = (float(lats[row_idx_j]), float(lons[row_idx_j]))
                            A[i, j] = get_invdistance(p1, p2) if i != j else float(1)
                        if obj_sim:
                            A_obj_single[i, :, j] = get_single_obj_count_similarity(obj_counts[i], obj_counts[j], obj_sim=obj_sim)
                if obj_sim:
                    A_obj_single /= float(np.amax(A_obj_single))
            elif A_type == 'none':
                A = np.random.rand(max_no_imgs, max_no_imgs)
            else:
                raise
            A = np.expand_dims(A, axis=1)
            if obj_sim:
                A = np.concatenate((A, A_obj_single), axis=1)
        clusters_A_test.append(A)
        cnt += 1

    train_sz = np.array(clusters_V).shape[0]
    full_sz = np.append(np.array(clusters_V), np.array(clusters_V_test), axis=0).shape[0]
    
    # concatenate train & test on return
    return np.append(np.array(clusters_V), np.array(clusters_V_test), axis=0), np.append(np.array(clusters_A), np.array(clusters_A_test), axis=0), np.append(np.array(np.reshape(labels, [-1])), np.array(np.reshape(labels_test, [-1])), axis=0), train_sz, full_sz
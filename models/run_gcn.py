import argparse
import graphcnn.setup.cluster as sc
from graphcnn.experiment_cluster import *
import pickle as pkl
import numpy as np
from pathlib import Path
import sys
import tensorflow.compat.v1 as tf
import joblib

# Uncomment to check for GPU
# print("========================================")
# print(tf.compat.v1.ConfigProto(log_device_placement=True))
# print("Num devices Available: ", len(tf.config.list_physical_devices()))
# print("=")
# print(tf.config.list_physical_devices())
# print("=")
# tf.test.is_gpu_available()
# print("========================================")

parser = argparse.ArgumentParser()
parser.add_argument('--vertices_fn', default='vertices_ex_pov.csv', help='Name of csv file to save vertices')
parser.add_argument('--adj_fn', default='adjacencies_ex_pov.csv', help='Name of csv file to save adj matrix')
parser.add_argument('--target', default='pov_label', help='Name of target var (pov_label, pop_label, bmi_label, pov, pop, bmi)')
parser.add_argument('--img_csv', default='.', help='Name of csv file that holds image-level data')
parser.add_argument('--train_val_dir', default='.', help='Name of directory that holds train/val cluster csvs')
parser.add_argument('--pretrained_image_file', default='.', help='Name of h5py file that holds pretrained 1-D image representations')
parser.add_argument('--A_type', default='inv', choices=['inv', 'none'], help='How you would like A constructed. Choose between inv, or none (random)')
parser.add_argument('--obj_sim', default=None, choices=['cosine','jaccard',None], help='Use jaccard or cosine similarity on object counts if specified. When None, do not use object counts.')
parser.add_argument('--cluster_limit', default=1, help='minimum limit for images per cluster')
parser.add_argument('--V_type', default='feats', choices=['feats','obj', 'both'], help='What to include in V. Choose between feats, obj, or both.')
parser.add_argument('--slice_A', default=False, choices=['True', 'False'], help='Only use if your A contains single shared object counts. Slices for top 5 objects relative to each label.')
parser.add_argument('--lr', default=0.000001, help='Learning rate')
parser.add_argument('--batch_size', default='256', help='Batch size')
parser.add_argument('--num_iter', default='3000', help='Number of training/testing iterations')

args = parser.parse_args()


target = args.target

is_regression = False
if "_label" in target and ("pov" in target or "bmi" in target or "pop" in target):
    is_regression = False
elif "pov" == target or "bmi" == target or "pop" == target:
    is_regression = True
else:
    raise

train_val_dir = args.train_val_dir if args.train_val_dir[-1] == '/' or args.train_val_dir[-1] == '\\' else args.train_val_dir + '/'
A_type = str(args.A_type).lower()
obj_sim = args.obj_sim if args.obj_sim else "none"
fn_cluster_limit = args.cluster_limit if int(args.cluster_limit) != -1 else "none"

print("Collecting data...")
graph_pkl_fn = "PT={}_TARGET={}_A={}_meters_V={}_obj_sim={}_cluster_limit={}.pkl".format(args.pretrained_image_file.split('/')[-1].split('.')[0], target, A_type, args.V_type, obj_sim, fn_cluster_limit)
graph_pkl_fp = Path(graph_pkl_fn)
dataset = None
if not graph_pkl_fp.is_file():
    print("extracting graph data...")
    dataset = sc.load_cluster_dataset(target, args.img_csv, train_val_dir, args.pretrained_image_file, 
                                      V_type=args.V_type, A_type=A_type, obj_sim=args.obj_sim, 
                                      CLUSTER_LIMIT=int(args.cluster_limit))
    with open(graph_pkl_fn, 'wb') as f:
        #pkl.dump(dataset, f, protocol=pkl.HIGHEST_PROTOCOL)
        joblib.dump(dataset, f)
else:
    print("loading presaved graph data...")
    with open(graph_pkl_fn, 'rb') as f:
        dataset = joblib.load(f)
        
GraphCNNGlobal.BN_DECAY = 0.3

class MapillaryExperiment(object):
    def create_network(self, net, input):
        net.create_network(input)
        net.make_graphcnn_layer(64)
        net.make_graphcnn_layer(64)
        net.make_graph_embed_pooling(no_vertices=32)
            
        net.make_graphcnn_layer(32)
        net.make_graphcnn_layer(32)
        
        net.make_graph_embed_pooling(no_vertices=8)
            
        net.make_fc_layer(256)
        
        if is_regression:
            net.make_fc_layer(1, name='final', with_bn=False, with_act_func = False)
        else:
            net.make_fc_layer(2, name='final', with_bn=False, with_act_func = False)
        
print("Building model...")
exp = GraphCNNExperiment('Mapillary', 'mapillary', MapillaryExperiment())

exp.num_iterations = int(args.num_iter)
exp.starter_learning_rate = float(args.lr)
exp.learning_rate_step = 200
exp.learning_rate_exp = 0
exp.train_batch_size = int(args.batch_size)
exp.test_batch_size = 0
exp.optimizer = 'adam'
exp.debug = True

exp.preprocess_data(dataset, is_regression)
del dataset

acc, std = exp.run_kfold_experiments(is_regression, no_folds=1)
print_ext('Result: %.2f (+- %.2f)' % (acc, std))

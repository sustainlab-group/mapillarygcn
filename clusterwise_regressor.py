from keras.preprocessing import image
from keras.applications import resnet50, densenet
from keras.models import Model, model_from_json, load_model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils import np_utils
from PIL import Image
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, CSVLogger
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
import tensorflow as tf

import argparse
import numpy as np
import pandas as pd
import glob
import math
import random
import os
import pdb
import copy
import json

parser = argparse.ArgumentParser(
    description="Train a baseline model that predicts label based onobject counts")
parser.add_argument("--data_csv", type=str,
                    default="/atlas/u/jihyeonlee/segment_mapillary/pipeline/povpopbmi_labels_per_image_224x224.csv")
parser.add_argument("--train_clusters_txt", type=str,
                    default="/atlas/u/jihyeonlee/segment_mapillary/pipeline/poverty/ia/train_clusters_ia.txt")
parser.add_argument("--val_clusters_txt", type=str,
                    default="/atlas/u/jihyeonlee/segment_mapillary/pipeline/poverty/ia/val_clusters_ia.txt")
parser.add_argument("--label_type", type=str, default="pov")

parser.add_argument("--model_save_dir", type=str,
                    default="/atlas/u/jihyeonlee/segment_mapillary/pipeline/models")
parser.add_argument("--model_save_name_stem", type=str,
                    default="mp_clswise_det_pov")

parser.add_argument("--eval_only", action='store_true')

parser.add_argument("--feature_type", type=str,
                    default="cluster", choices=["cluster", "img"])

parser.add_argument("--min_cluster_size", type=int, default=0)

# Args related to training and model hyperparameters
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--momentum", type=float, default=0.9)


# num_stuff == 28, num_thing == 37, 255 is void class + 1 for total number of images
NUM_FEATS = 66 
#idx_df = pd.read_csv('ia_mapillary_hh_wealthindex_agg.csv')

def batch_generator(train_df, num_classes, args, features=None, labels=None, batch_size=512):
    while True:
        for batch_index in range(round(features.shape[0] / batch_size)):
            x_train = features[batch_index *
                               batch_size:batch_index * batch_size + batch_size]
            y_train = labels[batch_index *
                             batch_size:batch_index * batch_size + batch_size]
            yield x_train, y_train


def learning_rate_scheduler(epoch, lr=1e-3):
    #if epoch > 50:
    #    lr *= 1e-1
    # if epoch > 180:
    #     lr *= 0.5e-3
    # elif epoch > 120:
    #     lr *= 1e-2
    # elif epoch > 80:
    #     lr *= 1e-1
    # elif epoch > 50:
    #     lr *= 1e-1
    print("Set Learning Rate : {}".format(lr))
    return lr


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def pearson_cc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def main(args):
    df = pd.read_csv(args.data_csv)

    with open(args.train_clusters_txt, 'r') as f:
        train_clusters = set([x.strip() for x in f.readlines()])
    with open(args.val_clusters_txt, 'r') as f:
        val_clusters = set([x.strip() for x in f.readlines()])

    # ==================================
    train_df = df.loc[df['unique_cluster'].isin(train_clusters)]
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = df.loc[df['unique_cluster'].isin(val_clusters)]
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    num_classes = 1  # value of index
    # =================================

    # =================================
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(NUM_FEATS,)))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes))

    model.compile(loss='mse',
                  optimizer=Adam(lr=learning_rate_scheduler(0, lr=args.lr)),
                  metrics=['mse', 'mae',  r2])
    # =================================

    # =================================
    model_file = os.path.join(
        args.model_save_dir, args.label_type + '_' + args.feature_type + '.hdf5')
    log_file = os.path.join(args.model_save_dir,
                            args.label_type + '_' + args.feature_type + '.log')
    checkpoint = ModelCheckpoint(model_file, monitor='val_acc',
                                 save_best_only=True, save_weights_only=False, mode='max', period=1)
    csv_logger = CSVLogger(log_file)
    callbacks_list = [checkpoint, csv_logger]
    # =================================

    # Train Clusters Data
    print("Loading train clusters data")
    train_tensor = np.load('features/trainpov_feats.npy')
    train_labels = np.load('features/trainpov_labels.npy')

    # Val Clusters Data
    print("Loading val clusters data")
    val_tensor = np.load('features/valpov_feats.npy')
    val_labels = np.load('features/valpov_labels.npy')

    if args.min_cluster_size > 0:
        print("Minimum cluster size is at least {} images".format(
            args.min_cluster_size))
        idx_to_use = train_tensor[:, -1] > args.min_cluster_size
        train_tensor = train_tensor[idx_to_use]
        train_labels = train_labels[idx_to_use]
        idx_to_use = val_tensor[:, -1] > args.min_cluster_size
        val_tensor = val_tensor[idx_to_use]
        val_labels = val_labels[idx_to_use]
        print("Training data size:  {}".format(train_tensor.shape[0]))
        print("Val data size: {}".format(val_tensor.shape[0]))

    model.fit_generator(batch_generator(train_df, num_classes, args, features=train_tensor, labels=train_labels, batch_size=args.batch_size), steps_per_epoch=round(len(train_df) / args.batch_size) - 1, epochs=args.num_epochs,
                        validation_data=batch_generator(val_df, num_classes, args, features=val_tensor, labels=val_labels, batch_size=args.batch_size), validation_steps=round(len(val_df) / args.batch_size) - 1, callbacks=callbacks_list)


if __name__ == "__main__":
    main(parser.parse_args())


import argparse
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='random', choices=['random', 'avg'], help='Baseline to run. Classification: Random or Average of Neighbors. Regression only supports Random.')
parser.add_argument('--label', default='pov_label', help='Which index to use')
parser.add_argument('--num_neighbors', default=1000, help='Number of neighboring clusters to average')
args = parser.parse_args()


df = pd.read_csv('data.csv')

# Read in cluster lists
train_clusters = [line.rstrip('\n') for line in open("train_clusters_ia.txt")]
val_clusters = [line.rstrip('\n') for line in open("val_clusters_ia.txt")]

def prepare_data():
    # Split into train vs val
    train_df = df.loc[ df['unique_cluster'].isin(train_clusters) ]
    # emulating training on districts by dividing into north and south (25 deg)
#    train_df = df.loc[df['lat'] >= 23]  
    val_df = df.loc[ df['unique_cluster'].isin(val_clusters) ]
    print(train_df.shape)

    def get_x_y(df):
        lat = df['lat']
        lon = df['lon']
        x = [list(a) for a in zip(lat, lon)]
        y = df[args.label].to_numpy()
        return x, y

    train_x, train_y = get_x_y(train_df)
    val_x, val_y = get_x_y(val_df)
    return train_x, train_y, val_x, val_y


if 'label' in args.label: # Classification task
    if args.baseline == 'random':
        labels = []
        for cluster in val_clusters:
            cluster_imgs = df.loc[df['unique_cluster'] == cluster]
            if cluster_imgs.shape[0] > 0:
                label = cluster_imgs[args.label].values[0]
                labels.append(label)
        preds = np.array([random.randint(0, 1) for i in range(len(labels))])
        correct = np.sum(preds == labels)
        print(float(correct) / len(val_clusters))
    else:   
        train_x, train_y, val_x, val_y = prepare_data()
        nbrs = KNeighborsClassifier(n_neighbors=args.num_neighbors)
        nbrs.fit(train_x, train_y)
        print(nbrs.score(val_x, val_y))
else: # Regression task
    train_x, train_y, val_x, val_y = prepare_data()
    nbrs = KNeighborsRegressor(n_neighbors=args.num_neighbors)
    nbrs.fit(train_x, train_y)
    print(nbrs.score(val_x, val_y))

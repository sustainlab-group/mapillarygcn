import argparse
import logging
import json
import numpy as np
import pandas as pd

from scipy.stats.stats import pearsonr

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='features/', help='Base name to save model and log file')
parser.add_argument('--log_file', default='results.log', help='File to save results from classifiers')
parser.add_argument('--train_saved_feats', default=None, help='If features were already saved, specify file')
parser.add_argument('--train_saved_labels', default=None, help='If labels were already saved, specify file')
parser.add_argument('--val_saved_feats', default=None, help='If features were already saved, specify file')
parser.add_argument('--val_saved_labels', default=None, help='If labels were already saved, specify file')
parser.add_argument('--label', default='pov_label', help='Which index to use')
args = parser.parse_args()

logging.basicConfig(filename=args.log_file, level=logging.DEBUG)


def get_cluster_features(df, clusters, counts, train=True):
    cluster_features = []
    cluster_labels = np.array([])
    cluster_idxs = df['unique_cluster'].to_numpy()
    for i, cluster_id in enumerate(clusters):
        # Get the features for this cluster
        imgs_in_cluster = np.argwhere(cluster_idxs == cluster_id).flatten()
        cluster_size = len(imgs_in_cluster)
        if cluster_size == 0:
            continue
        feats = counts[imgs_in_cluster, :]
        feats = np.sum(feats, axis=0) # sum up the object counts from each image

        # Add a feature for size of cluster
        feats = np.append(feats, [cluster_size])

        # Append this cluster's features to the rest
        cluster_features.append(feats)

        # Append this cluster's label
        target = df[args.label].iloc[imgs_in_cluster[0]]
        cluster_labels = np.append(cluster_labels, target)

    cluster_features = np.array(cluster_features)
    print(cluster_features.shape)
    print(cluster_labels.shape)

    # Save them
    prefix = "train" if train else "val"
    saved_feats = args.save_dir + prefix + args.label + '_feats.npy'
    saved_labels = args.save_dir + prefix + args.label + '_labels.npy'
    np.save(saved_feats, cluster_features)
    np.save(saved_labels, cluster_labels)
    return saved_feats, saved_labels
    

def process_objects(objects): 
    NUM_FEATS = 66 # num_stuff == 28, num_thing == 37, 255 is void class
    try:
        objects = json.loads(objects)
        objects = [NUM_FEATS - 1 if x == 255 else x for x in objects]
        # turns instances into histogram of counts
        counts = np.histogram(objects, bins=np.arange(0, NUM_FEATS))[0] 
        exit
    except:
        counts = np.zeros(NUM_FEATS)
    return counts


def get_class_results(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    score = metrics.accuracy_score(y_val, pred)
    print("accuracy:   %0.4f" % score)
    print(metrics.classification_report(y_val, pred, target_names=['low', 'high']))
    print(metrics.confusion_matrix(y_val, pred))
    logging.info("accuracy:   %0.4f" % score)
    logging.info(metrics.classification_report(y_val, pred, target_names=['low', 'high']))
    logging.info(metrics.confusion_matrix(y_val, pred))


def get_reg_results(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    mse = metrics.mean_squared_error(y_val, pred)
    r2 = metrics.r2_score(y_val, pred)
    pear = pearsonr(y_val.ravel(), pred)[0]
    print('Mean squared error: %.4f' % mse)
    print('R2 (coefficient of determination): %.4f' % r2)
    print('PearsonR: %.4f' % pear)
    logging.info('Mean squared error: %.4f' % mse)
    logging.info('R2 (coefficient of determination): %.4f' % r2)
    logging.info('PearsonR: %.4f' % pear)


def run_classifiers(X_train, y_train, X_val, y_val):
    for clf, name in (
            (KNeighborsClassifier(n_neighbors=3), "kNN"),
            (RandomForestClassifier(n_estimators=100, max_depth=None), "Random forest 100"),
            (RandomForestClassifier(n_estimators=300, max_depth=None), "Random forest 300"),
            (GradientBoostingClassifier(learning_rate=1e-1, n_estimators=100, max_depth=7), "GBDT 7"),
            (GradientBoostingClassifier(learning_rate=1e-1, n_estimators=100, max_depth=10), "GBDT 10"),
            ):
        print('=' * 80)
        print(name)
        logging.info('=' * 80)
        logging.info(name)
        get_class_results(clf, X_train, y_train, X_val, y_val)    


def run_regressors(X_train, y_train, X_val, y_val):
    for clf, name in (
            (KNeighborsRegressor(n_neighbors=3), "kNN"),
            (RandomForestRegressor(n_estimators=100, max_depth=None), "Random forest 100"),
            (RandomForestRegressor(n_estimators=300, max_depth=None), "Random forest 300"),
            (GradientBoostingRegressor(learning_rate=1e-1, n_estimators=300, max_depth=5), "GBDT 5"),
            (GradientBoostingRegressor(learning_rate=1e-1, n_estimators=300, max_depth=7), "GBDT 7"),
            ):
        print('=' * 80)
        print(name)
        logging.info('=' * 80)
        logging.info(name)
        get_reg_results(clf, X_train, y_train, X_val, y_val) 


def main():
    df = pd.read_csv('data.csv')
    objects = df['features'].to_numpy()  # object instances
    counts = np.array(list(map(process_objects, objects))) # converted to counts

    if not args.train_saved_feats:
        train_f = open('train_clusters_ia.txt', 'r')
        val_f = open('val_clusters_ia.txt', 'r')
        train_clusters = [x[:-1] for x in train_f.readlines()]
        val_clusters = [x[:-1] for x in val_f.readlines()]
        train_f.close()
        val_f.close()
        args.train_saved_feats, args.train_saved_labels = get_cluster_features(df, train_clusters, counts)
        args.val_saved_feats, args.val_saved_labels = get_cluster_features(df, val_clusters, counts, train=False)
    
    X_train, y_train = np.load(args.train_saved_feats), np.load(args.train_saved_labels)
    X_val, y_val = np.load(args.val_saved_feats), np.load(args.val_saved_labels)

    if 'label' in args.label:
        run_classifiers(X_train, y_train, X_val, y_val)
    else:
        run_regressors(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main() 

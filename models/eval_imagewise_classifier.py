import argparse
import logging
import os
import h5py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from collections import Counter
from skimage import io, transform
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, default='results_pov_label_img.log', help='Log file')
parser.add_argument('--model_weights', type=str, default='models/pov_classify', help='Where pretrained model is located')
parser.add_argument('--resnet_ver', type=str, default='resnet34', help='Which ResNet architecture was used')
parser.add_argument('--label', type=str, default='pov_label', help='Label')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

logging.basicConfig(filename=args.log_file,level=logging.DEBUG)


class ClusterImgDataset(Dataset):
    def __init__(self, df, device):
        self.img_paths = df['img_path_224x224'].to_numpy()
        self.device = device

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        image = io.imread(self.img_paths[idx])
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1)
        return image_tensor


def create_model():
    pretrained_weights = torch.load(args.model_weights)
    if args.resnet_ver == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif args.resnet_ver == 'resnet34':
        model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(pretrained_weights)
    return model


def get_majority_vote(cluster_dataset, model, device):
    model.eval()
    pred = np.array([])
    generator = DataLoader(cluster_dataset, batch_size=args.batch_size, num_workers=1)
    for batch in generator:
        batch = batch.to(device, dtype=torch.float32)
        output = model(batch)
        predicted = output.argmax(dim=1, keepdim=True)
        pred = np.append(pred, predicted.cpu().numpy())
        del batch
    del generator
    votes = Counter(pred)
    majority = votes.most_common(1)[0][0]
    del pred
    return majority

    
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv('data.csv')

    val_f = open('val_clusters_ia.txt', 'r')
    val_clusters = [x[:-1] for x in val_f.readlines()]
    val_f.close()

    print("Creating model...")
    model = create_model().to(device)
    model.eval()

    val_c = []
    y_true = np.array([])
    y_pred = np.array([])

    for cluster in tqdm(val_clusters):
        cluster_df = df.loc[df['unique_cluster'] == cluster]
        if cluster_df.shape[0] == 0:
            continue
        val_c.append(cluster)
        target = cluster_df[args.label].values[0]
        y_true = np.append(y_true, cluster_df[args.label].values[0])
        dataset = ClusterImgDataset(cluster_df, device)
        y_pred = np.append(y_pred, get_majority_vote(dataset, model, device))
        del cluster_df, dataset

    logging.debug(y_true.shape)
    logging.debug(y_pred.shape)
    print(confusion_matrix(y_true, y_pred))
    logging.debug(confusion_matrix(y_true, y_pred))
    prec, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print("Precision: {}\nRecall: {}\nF-Score:  {}\nSupport: {}".format(prec, recall, fscore, support))
    logging.debug("Precision: {}\nRecall: {}\nF-Score:  {}\nSupport: {}".format(prec, recall, fscore, support))

    # Save predictions
    df = pd.DataFrame({'unique_cluster': np.array(val_c),
                   args.label: y_true,
                   args.label + 'pred': y_pred})
    df.to_csv(args.label + '_preds.csv', index=False)
    

if __name__ == "__main__":
    main() 
    

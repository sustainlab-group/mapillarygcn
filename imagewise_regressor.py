import argparse
import logging
import os

import h5py
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import io, transform
from sklearn.metrics import r2_score


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet34', choices=['resnet18', 'resnet34'], help='ResNet Version')
parser.add_argument('--save_name', default='models/reg_pov', help='Base dir name to keep saved model and log')
parser.add_argument('--label', default='pov', choices=['pov', 'pop', 'bmi'], help='Label used for training')
parser.add_argument('--lr', default=1e-3, help='Training learning rate')
parser.add_argument('--batch_size', default=256, help='Training batch size')
parser.add_argument('--num_epochs', default=100, help='Training number of epochs')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained Imagenet weights')
parser.add_argument('--eval_mode', action='store_true', help='Only evaluates')
args = parser.parse_args()

if not os.path.exists(args.save_name):
    os.makedirs(args.save_name)
logging.basicConfig(filename=args.save_name + '/log', level=logging.DEBUG)
writer = SummaryWriter(args.save_name)


class ImgDataset(Dataset):
    def __init__(self, df, device):
        self.img_paths = df['img_path_224x224'].to_numpy()
        self.targets = df[args.label].to_numpy()
        self.device = device

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, idx):
        image = io.imread(self.img_paths[idx])
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2,0,1).float()
        target = torch.Tensor(np.array([self.targets[idx]]))
        return image_tensor, target


def create_model():
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    model.fc = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
    return model


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        target = target.cpu().detach().numpy()
        pred = output.cpu().detach().numpy()
        r2 = r2_score(target, pred)
        
        if batch_idx % 5 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR2: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 
                r2))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR2: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 
                r2))
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('R2/train', r2, epoch * len(train_loader) + batch_idx)
   

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            total += target.size(0)
            output = model(data)
            target = target.cpu().detach().numpy().squeeze()
            pred = output.cpu().detach().numpy().squeeze()
            y_true = np.append(y_true, target)
            y_pred = np.append(y_pred, pred)
    r2 = r2_score(y_true, y_pred)

    logging.info('\nTest set: R2: {:.4f}\n'.format(r2)) 
    print('\nTest set: R2: {:.4f}\n'.format(r2))
    writer.add_scalar('R2/test', r2, epoch * total)
    return r2, y_true, y_pred


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Getting the clusters')
    train_f = open('train_clusters_ia.txt', 'r')
    val_f = open('val_clusters_ia.txt', 'r')
    train_clusters = [x[:-1] for x in train_f.readlines()]
    val_clusters = [x[:-1] for x in val_f.readlines()]
    train_f.close()
    val_f.close()

    print('Preparing the dataloader')
    df = pd.read_csv('data.csv')
    train_df = df.loc[df['unique_cluster'].isin(train_clusters)]
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = df.loc[df['unique_cluster'].isin(val_clusters)]
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    train_dataset = ImgDataset(train_df, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_dataset = ImgDataset(val_df, device)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    model = create_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    if not args.eval_mode:
        print('Starting training')
        best_r2 = 0.
        for epoch in range(1, args.num_epochs+1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            r2, y_true, y_pred = test(model, device, val_loader, criterion, epoch)
            if r2 >= best_r2:
                best_r2 = r2
                torch.save(model.state_dict(), args.save_name + "/model")
                logging.info("\nSaved model with R2: {:.4f}\n".format(best_r2))
        
        logging.info("\nBest R2: {:.4f}\n".format(best_r2))
        print("\nBest R2: {:.4f}\n".format(best_r2))
    else:
        r2, y_true, y_pred = test(model, device, val_loader, criterion, 1)
        print("\nVal R2: {:.4f}\n".format(best_r2))

    # Saves the predictions
    df = pd.DataFrame({'unique_cluster': val_clusters,
                   args.label: y_true,
                   args.label + 'pred': y_pred})
    df.to_csv(args.save_name + '/' + args.label + '_preds.csv', index=False)

if __name__ == "__main__":
    main() 
    

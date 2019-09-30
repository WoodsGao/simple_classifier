import cv2
import numpy as np
import os
from cv_utils.processors import *
from cv_utils.utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, features):
        """MLP

        Arguments:
            features {list} -- int numbers for each layer e.g.[1000, 2000, 1000, 8]
        """
        super(MLP, self).__init__()
        layers = []
        for fi, feature_num in enumerate(features[:-1]):
            if fi+2 == len(features):
                layer = nn.Linear(features[fi], features[fi+1])
            else:
                layer = nn.Sequential(
                    nn.Linear(features[fi], features[fi+1], bias=False),
                    nn.Dropout(0.5),
                    nn.BatchNorm1d(features[fi+1]), nn.LeakyReLU())
            layers.append(layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


def train_mlp(train_dir, val_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_loss = 1000
    best_acc = 0
    processor_list = [
        ComputeHog(),
        SobelX(),
        SobelY(),
        Resize(size=(16, 16))
    ]
    train_inputs, train_targets = simple_dataloader(train_dir,
                                                    processor_list=processor_list
                                                    )
    train_inputs = torch.from_numpy(train_inputs).to(device)
    train_targets = torch.from_numpy(train_targets).to(device)
    val_inputs, val_targets = simple_dataloader(val_dir,
                                                processor_list=processor_list
                                                )
    val_inputs = torch.from_numpy(val_inputs).to(device)
    val_targets = torch.from_numpy(val_targets).to(device)
    model = MLP([train_inputs.size(1), 100, 1000,
                 int(max(train_targets.max(), val_targets.max()))+1])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    pbar = tqdm(range(100))
    for epoch in pbar:
        # train
        model.train()
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # validate
        model.eval()
        with torch.no_grad():
            outputs = model(val_inputs)
            loss = criterion(outputs, val_targets)
            val_loss = loss.item()
            val_acc = float(
                (outputs.max(1)[1] == val_targets).sum()) / val_targets.size(0)
        state_dict = {
            'model': model.state_dict(),
            'acc': val_acc,
            'loss': val_loss,
            'epoch': epoch
        }
        if val_loss < best_loss:
            torch.save(state_dict, 'best_loss.pt')
        if val_acc > best_acc:
            torch.save(state_dict, 'best_acc.pt')
        pbar.set_description('eopch: %5g    train_loss: %10g    val_loss: %10g    val_acc: %5g'
                             % (epoch, train_loss, val_loss, val_acc))


if __name__ == "__main__":
    train_dir = 'data/road_mark/train'
    val_dir = 'data/road_mark/val'
    train_mlp(train_dir, val_dir)

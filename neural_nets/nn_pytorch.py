'''
    Matt Briones
    Last Edited: March 8, 2020
'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def run(train, labels, test):
    print('do pytorch stuff now')

def get_datasets(filepath, train_ratio = 0.80):
    data = pd.read_csv(filepath)
    train_data = data.sample(frac = train_ratio)
    heldout_data = data.loc[~data.index.isin(train_data.index)]
    return train_data, heldout_data

train_df, heldout_df = get_datasets('../data/train.csv')

trainSet = torch.utils.data.DataLoader(train_df, batch_size = 10, shuffle = True)
heldoutSet = torch.utils.data.DataLoader(heldout_df, batch_size = 10, shuffle = True)

class TweetNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hl1 = nn.Linear(input_size, 3)
        self.hl2 = nn.Linear(3, 3)
        self.hl3 = nn.Linear(3, 3)
        self.hl4 = nn.Linear(3, output_size)

    def forward(self, x):
        x = F.relu(self.hl1)
        x = F.relu(self.hl2)
        x = F.relu(self.hl3)
        x = self.hl4
        return x

tNet = TweetNet(3, 3) #temporary values

def train(trainset, epochs = 3, net -> TweetNet):
    optimizer = optim.Adam(net.paramters(), lr = 1e-5) #Adam algorithm
    for epoch in range(epochs):
        for data in trainset:
            X, y = data
            tNet.zero_grad()
            output = tNet(X.view(-1, 3)) # 3 is a fill in value
            loss = F.binary_cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            

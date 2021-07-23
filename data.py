# -*- coding: utf-8 -*-

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import random

class MyDataset(Dataset):
    def __init__(self, data_path, mode, fmt, zoom=1.0, train_disc = 1.0, size=128):
        self.size = size
        self.zoom = zoom
        
        with open(os.path.join(data_path, 'data.txt'), 'r') as f:
            lst, acts = [], []
            for l in f.readlines():
                nm = l.split()[0].split('_') # e.g. ['000001', '03']
                a = l.split()[1]
                lst.append(nm)
                acts.append(int(a))
            trips, act_trips = [], []
            for i in range(len(lst) - 2):
                if lst[i][0] == lst[i+1][0] and lst[i+1][0] == lst[i+2][0]:
                    trips.append((lst[i][0] + '_' + lst[i][1], lst[i+1][0] + '_' + lst[i+1][1], \
                                  lst[i+2][0] + '_' + lst[i+2][1]))
                    act_trips.append((acts[i], acts[i+1], acts[i+2]))
        
        ds = list(zip(trips, act_trips))

        n_train = int(0.9 * len(ds)) # train: test = 9:1
        random.seed(3) # fixed train data and test data!
        random.shuffle(ds)
        if mode == 'train':
            ds = ds[0: int(n_train * train_disc)]
        elif mode == 'test':
            ds = ds[n_train: ]
        else:
            raise NotImplementedError;
            
        
        self.data = []
        self.actions = []
        for p, a in ds:
            self.data.append((os.path.join(data_path, p[0] + fmt), os.path.join(data_path, p[1] + fmt),\
                              os.path.join(data_path, p[2] + fmt)))
            self.actions.append(a)
        #print(self.data)
        #print(self.actions)

    def __getitem__(self, index):
        #zoom
        cut = int(self.size * (self.zoom - 1.0) / 2)
        
        frame0 = Image.open(self.data[index][0])
        frame0 = np.asarray(frame0.resize((self.size, self.size)).crop((cut, cut, self.size-cut, self.size-cut)).resize((self.size, self.size)), \
                            dtype=np.float32) / 127.5 - 1.0
        frame0 = torch.from_numpy(np.transpose(frame0, (2, 0, 1)))
        
        frame1 = Image.open(self.data[index][1])
        frame1 = np.asarray(frame1.resize((self.size, self.size)).crop((cut, cut, self.size-cut, self.size-cut)).resize((self.size, self.size)), \
                            dtype=np.float32) / 127.5 - 1.0
        frame1 = torch.from_numpy(np.transpose(frame1, (2, 0, 1)))
        
        frame2 = Image.open(self.data[index][2])
        frame2 = np.asarray(frame2.resize((self.size, self.size)).crop((cut, cut, self.size-cut, self.size-cut)).resize((self.size, self.size)), \
                            dtype=np.float32) / 127.5 - 1.0
        frame2 = torch.from_numpy(np.transpose(frame2, (2, 0, 1)))
        
        return (frame0, frame1, frame2), self.actions[index]

    def __len__(self):
        return len(self.data)

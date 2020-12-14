import os
import sys
import pickle
import torch
import numpy as np
import torch.utils.data as data
from convlab2.util.multiwoz.state import default_state
from convlab2.policy.vector.dataset import ActDataset

from torch.utils.data.dataset import Dataset
import json

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)
from convlab2.policy.vector.vector_multiwoz import MultiWozVector

class Dataset_MGAIL(Dataset):
    def __init__(self, dataset_path):
        self.state = []
        self.action = []
        self.goal = []
        
        domains = ['attraction', 'hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train']
        
        for path in dataset_path :
            with open(path) as f:
                records = json.load(f)
            
            for i in range(len(records['goal'])):
                g = []
                count = 0.
                for k in range(7):
                    if domains[k] in records['goal'][i]:
                        g.append(1.)
                        count += 1
                    else:
                        g.append(0.)
                for k in range(7):    
                    g[k] /= count
                for j in range(len(records['state'][i])):
                    self.state.append(np.array(records['state'][i][j]))
                    self.action.append(np.array(records['action'][i][j]))
                    self.goal.append(np.array(g))
        self.length = len(self.goal)
        print('number of data:', self.length)
        
    def __getitem__(self, index):
        return self.state[index], self.action[index], self.goal[index]
        
    def __len__(self):
        return self.length


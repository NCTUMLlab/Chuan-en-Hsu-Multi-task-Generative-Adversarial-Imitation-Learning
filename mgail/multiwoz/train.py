import os
import torch
import logging
import json
import sys
import argparse
from torch import nn
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from convlab2.policy.rlmodule import DiscretePolicy, MultiDiscretePolicy, Discriminator, Classifier
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.policy.mgail.train import MGAIL_Trainer_Abstract
from convlab2.policy.mgail.multiwoz.loader import ActMLEPolicyDataLoaderMultiWoz
from convlab2.util.train_util import init_logging_handler

from convlab2.policy.mgail.multiwoz.mgail import MGAIL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MGAIL_Trainer(MGAIL_Trainer_Abstract):
    def __init__(self, cfg, load_path):
        self._init_data(cfg, load_path)
        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)       
        
        # override the loss defined in the MLE_Trainer_Abstract to support pos_weight
        #pos_weight = cfg['pos_weight'] * torch.ones(self.vector.da_dim).to(device=DEVICE)
        #self.multi_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.criterion = nn.BCELoss()
        self.policy = MGAIL()
        if load_path:
            self.policy.load(load_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default='', help="path of model to load")
    parser.add_argument("--config_path", type=str, default='config.json', help="path of configuration")
    args = parser.parse_args()
    
    #manager = ActMLEPolicyDataLoaderMultiWoz()
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    init_logging_handler(cfg['log_dir'])
    agent = MGAIL_Trainer(cfg, args.load_path)
    

    
    logging.debug('start training')
    
    best = float('inf')



    agent.set_dataloader()
    #print(eval(agent.records[0]))
    for e in range(cfg['epoch']):
        agent.imitating2(e)
        #best = agent.imit_test(e, best)
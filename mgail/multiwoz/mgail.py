# -*- coding: utf-8 -*-
import torch
import os
import json
from convlab2.policy.mgail.mgail import MGAILAbstract
from convlab2.policy.rlmodule import DiscretePolicy, MultiDiscretePolicy, Shared_policy, Shared_network, Classifier
from convlab2.policy.vector.vector_multiwoz import MultiWozVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "mle_policy_multiwoz.zip")

class MGAIL(MGAILAbstract):
    
    def __init__(self, path_config='config.json'):
        self.cfg = self.get_config(path_config)
        self.vector= self.get_vector()
        self.domains = ['attraction', 'hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train']
        ###########################
        self.policies = []
        self.policies_optim = []
        for i in range(7):
            self.policies.append(MultiDiscretePolicy(self.vector.state_dim, self.cfg['h_dim'], self.vector.da_dim).to(device=DEVICE))
            self.policies_optim.append(torch.optim.RMSprop(self.policies[i].parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay']))
        ##self.policy = Shared_policy(self.vector.state_dim, self.cfg['h_dim'], self.vector.da_dim).to(DEVICE)
        ##self.policy_optim = torch.optim.RMSprop(self.policy.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        ###########################    
        self.meta_policy = DiscretePolicy(self.vector.state_dim, self.cfg['h_dim'], 7).to(device=DEVICE)
        self.meta_policy_optim = torch.optim.RMSprop(self.meta_policy.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        ###########################
        self.network = Shared_network(self.vector.state_dim, self.cfg['h_dim'], self.vector.da_dim).to(DEVICE)
        self.network_optim = torch.optim.RMSprop(self.network.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        #self.discriminator = Discriminator(self.vector.state_dim, self.cfg['h_dim'], self.vector.da_dim).to(device=DEVICE)
        #self.discriminator_optim = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        ###########################
        #self.classifier = DiscretePolicy(self.vector.state_dim+self.vector.da_dim, self.cfg['h_dim'], 7).to(device=DEVICE)
        #self.classifier_optim = torch.optim.RMSprop(self.classifier.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        ###########################
        self.set_mode()
            
    def set_mode(self, train=False):
        if not train:
            for i in range(7):
                self.policies[i].eval()
            self.meta_policy.eval()
            #self.discriminator.eval()
            #self.classifier.eval()
            ##self.policy.eval()
            self.network.eval()
        else:            
            for i in range(7):
                self.policies[i].train()
            self.meta_policy.train()
            #self.discriminator.train()
            #self.classifier.train()
            ##self.policy.train()
            self.network.train()
             
    def get_config(self, path_config):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path_config), 'r') as f:
            return json.load(f) 
       
    def get_vector(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')       
        return MultiWozVector(voc_file, voc_opp_file)
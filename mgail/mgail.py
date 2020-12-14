# -*- coding: utf-8 -*-
import torch
import os
import zipfile
from convlab2.policy.policy import Policy
from convlab2.util.file_util import cached_path
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MGAILAbstract(Policy):

    def __init__(self, archive_file, model_file):
        self.vector = None
        self.policies = None
        self.policies_optim = None
        self.meta_policy = None
        self.meta_policy_optim = None
        #self.policy = None
        #self.policy_optim = None
        self.network = None
        self.network_optim = None
        #self.discriminator = None
        #self.discriminator_optim = None
        #self.classifier = None
        #self.classifier_optim = None

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        g = self.meta_policy.select_action(s_vec.to(device=DEVICE), True).cpu()
        
        a = self.policies[int(g[0])].select_action(s_vec.to(device=DEVICE), True).cpu()
        ##a = self.policy.select_action(s_vec.to(device=DEVICE), int(g[0]), True).cpu()
        action = self.vector.action_devectorize(a.detach().numpy())
        
        return action



    def init_session(self):
        """
        Restore after one session
        """
        pass



    def load(self, filename):
        print('load_mgail')
        path = filename + '_mgail'
        print(path)
        assert os.path.exists(path+'.meta.mdl'), 'could not find path:'+path
        self.meta_policy.load_state_dict(torch.load(path+'.meta.mdl', map_location=DEVICE))
        #self.discriminator.load_state_dict(torch.load(path+'.dis.mdl', map_location=DEVICE))
        #self.classifier.load_state_dict(torch.load(path+'.cla.mdl', map_location=DEVICE))
        for i in range(7):
            self.policies[i].load_state_dict(torch.load(path+'.pol_'+str(i)+'.mdl', map_location=DEVICE))
        ##self.policy.load_state_dict(torch.load(path+'.pol.mdl', map_location=DEVICE))
        self.network.load_state_dict(torch.load(path+'.net.mdl', map_location=DEVICE))
        logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(path))
        print('<<dialog policy>> loaded checkpoint from file: {}'.format(path))

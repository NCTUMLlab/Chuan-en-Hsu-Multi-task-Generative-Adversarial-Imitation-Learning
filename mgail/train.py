import os
import torch
import logging
import torch.nn as nn
#import sys
import json
#sys.path.append(root_dir)
from copy import deepcopy


from convlab2.util.train_util import to_device
from torch.utils.data import DataLoader
from convlab2.policy.mgail.dataset import Dataset_MGAIL
from convlab2.policy.vector.vector_multiwoz import MultiWozVector


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(a_weights, sample=True):
    a_probs = torch.sigmoid(a_weights)
    
    # [a_dim] => [a_dim, 2]
    #a_probs = a_probs.unsqueeze(1)
    #a_probs = torch.cat([1-a_probs, a_probs], 1)
    #a_probs = torch.clamp(a_probs, 1e-10, 1 - 1e-10)
    
    # [a_dim, 2] => [a_dim]
    #a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)
    
    return a_probs

class MGAIL_Trainer_Abstract():
    def __init__(self, cfg, load_path):
        self._init_data(cfg, load_path)
        self.policy = None
        ##Nightop
        self.vector = None
        
        
    def _init_data(self, cfg, load_path):
        self.batch_size = cfg['batchsz']
        #f = open('record.json', 'r')
        #self.data = json.load(f)
        #f.close()
        #self.data_train = manager.create_dataset('train', cfg['batchsz'])
        #self.data_valid = manager.create_dataset('val', cfg['batchsz'])
        #self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
    
        ##Nightop
        self.criterion = nn.BCELoss()
    
        
    def set_dataloader(self):
        #with open('/home/nightop/experiment/record_all.json') as f:
        #    self.records = json.load(f)
        #dataset = Dataset_MGAIL(self.records)
        #dataset_path = ['/home/nightop/experiment/dataset_attraction.json', '/home/nightop/experiment/dataset_train.json', '/home/nightop/experiment/record_all.json']
        dataset_path = ['/home/nightop/experiment/record_all.json']
        dataset = Dataset_MGAIL(dataset_path)
        self.num_minibatch = ((len(dataset)-len(dataset)%self.batch_size)/self.print_per_batch)
        self.dataloader = DataLoader(dataset, self.batch_size)
            
        
    ## Nightop    
    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy.policy(s)
        a = select_action(a_weights, False).float()
        #print(target_a[0])
        #a = self.policy.policy(s)
        #print(a[0])
        predict = self.discriminator(torch.cat((s, a), 1)).view(-1)
        label = torch.full((predict.size(0),), 1., device=DEVICE)
        #print(target_a[0])
        #print(a[0])
        #print(predict)
        loss_p = self.criterion(predict, label)
        #print(loss_p.item())
        return loss_p
    
    ## Nightop    
    def discriminator_loop(self, data):
        s, target_a = to_device(data)

        #real
        predict = self.discriminator(torch.cat((s, target_a), 1)).view(-1)
        label = torch.full((predict.size(0),), 1., device=DEVICE)     
        loss_r = self.criterion(predict, label)
        #fake
        a_weights = self.policy.policy(s)
        a = select_action(a_weights, False).float()
        #a = self.policy.policy(s)
        predict = self.discriminator(torch.cat((s, a), 1)).view(-1)
        label = torch.full((predict.size(0),), 0., device=DEVICE)
        loss_f = self.criterion(predict, label)
        return loss_r, loss_f
        
        
    def imitating(self, epoch):
        self.policy.policy.train()
        self.discriminator.train()
        p_loss = 0.
        d_loss = 0.
        ##Nightop
        for i, data in enumerate(self.data_train):
            #train discriminator
            self.discriminator.zero_grad()
            loss_r, loss_f = self.discriminator_loop(data)
            loss_d = loss_r + loss_f
            d_loss += loss_d.item()
            
            loss_r.backward()
            loss_f.backward()
            self.discriminator_optim.step()
            
            #train policy
            self.policy.policy.zero_grad()
            loss_p = self.policy_loop(data)
            p_loss += loss_p.item()
            loss_p.backward()
            self.policy.policy_optim.step()
              
            if (i+1) % (len(self.data_train)/self.print_per_batch) == 0:
                p_loss /= self.print_per_batch
                d_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_p:{}, loss_d:{}'.format(epoch, i, p_loss, d_loss))
                p_loss = 0.
                d_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.policy.eval()
        self.discriminator.eval()
    
    ####################################################################    
    def imitating2(self, epoch):
        self.policy.set_mode(True)
        p_loss = 0.
        d_loss = 0.
        m_loss = 0.
        c_loss = 0.
        ##Nightop
        for i, data in enumerate(self.dataloader):
            [state, action, goal] = data
 
            state = state.float().to(DEVICE)
            action = action.float().to(DEVICE)
            goal = goal.float().to(DEVICE)        
            g = goal.multinomial(1)
            #g = deepcopy(goal[0]).tolist().index(1.)
            #print(goal)
            #print(g)
            s_a = torch.cat((state, action), 1)
            

            #train meta_policy#########################
            self.policy.meta_policy.zero_grad()
            action_1 = self.policy.meta_policy(state)
            a_1 = select_action(action_1, False).float()
            #print(action_1)
            #print(torch.LongTensor([g, ]).to(DEVICE))
            loss_m = nn.CrossEntropyLoss()(a_1, torch.LongTensor([g, ]).to(DEVICE))
            loss_m.backward()
            m_loss += loss_m.item()
            self.policy.meta_policy_optim.step()
            #train discriminator########################################
            '''
            self.policy.discriminator.zero_grad()
            #real
            predict = self.policy.discriminator(s_a).view(-1)
            label = torch.full((predict.size(0),), 1., device=DEVICE)  
            loss_r = self.criterion(predict, label)
            #fake
            a_weights = self.policy.policies[g](state)
            a = select_action(a_weights, False).float()
            predict = self.policy.discriminator(torch.cat((state, a), 1)).view(-1)
            label = torch.full((predict.size(0),), 0., device=DEVICE)
            loss_f = self.criterion(predict, label)
            #update
            loss_d = loss_r + loss_f
            d_loss += loss_d.item()
            loss_r.backward()
            loss_f.backward()
            self.policy.discriminator_optim.step()
            '''

            self.policy.network.zero_grad()
            #real
            predict = self.policy.network(s_a, 'dis').view(-1)
            label = torch.full((predict.size(0),), 1., device=DEVICE)  
            loss_r = self.criterion(predict, label)
            #fake
            ##a_weights = self.policy.policy(state, g)
            a_weights = self.policy.policies[g](state)
            a = select_action(a_weights, False).float()
            predict = self.policy.network(torch.cat((state, a), 1), 'dis').view(-1)
            predict_c = self.policy.network(torch.cat((state, a), 1), 'cla')
            label = torch.full((predict.size(0),), 0., device=DEVICE)
            loss_f = self.criterion(predict, label)
            loss_c = nn.CrossEntropyLoss()(predict_c, torch.LongTensor([g, ]).to(DEVICE))
            #update
            loss_d = loss_r + loss_f
            c_loss += loss_c.item()
            d_loss += loss_d.item()
            loss_r.backward(retain_graph=True)
            loss_f.backward(retain_graph=True)
            loss_c.backward(retain_graph=True)
            self.policy.network_optim.step()
            
            
            
            
            #train policy###################################
            
            self.policy.policies[g].zero_grad()
            
            a_weights = self.policy.policies[g](state)
            a = select_action(a_weights, False).float()
            #predict = self.policy.discriminator(torch.cat((state, a), 1)).view(-1)
            predict = self.policy.network(torch.cat((state, a), 1), 'dis').view(-1)
            predict_c = self.policy.network(torch.cat((state, a), 1), 'cla')
            label = torch.full((predict.size(0),), 1., device=DEVICE)
            loss_pd = self.criterion(predict, label)
            loss_pc = nn.CrossEntropyLoss()(predict_c, torch.LongTensor([g, ]).to(DEVICE))
            #update
            loss_p = loss_pd + loss_pc
            p_loss += loss_p.item()
            loss_pd.backward(retain_graph=True)
            loss_pc.backward(retain_graph=True)
            #loss_p.backward(retain_graph=True)
            self.policy.policies_optim[g].step()
            
            '''
            self.policy.policy.zero_grad()
            
            a_weights = self.policy.policy(state, g)
            a = select_action(a_weights, False).float()
            predict = self.policy.network(torch.cat((state, a), 1), 'dis').view(-1)
            predict_c = self.policy.network(torch.cat((state, a), 1), 'cla')
            label = torch.full((predict.size(0),), 1., device=DEVICE)
            loss_pd = self.criterion(predict, label)
            loss_pc = nn.CrossEntropyLoss()(predict_c, torch.LongTensor([g, ]).to(DEVICE))
            #update
            loss_p = loss_pd + loss_pc
            p_loss += loss_p.item()
            loss_pd.backward(retain_graph=True)
            loss_pc.backward(retain_graph=True)
            self.policy.policy_optim.step()
            '''
            #train classifier##################
            '''
            self.policy.policies[g].zero_grad()
            self.policy.classifier.zero_grad()
            
            output_c = self.policy.classifier(torch.cat((state, a), 1))
            #print(output_c)
            
            loss_c = nn.CrossEntropyLoss()(output_c, torch.LongTensor([g, ]).to(DEVICE))
            
            c_loss += loss_c.item()
            loss_c.backward()
            self.policy.policies_optim[g].step()
            self.policy.classifier_optim.step()
            '''


            n = int(self.num_minibatch/self.print_per_batch)
            if (i+1) % n == 0:
                p_loss /= n
                d_loss /= n
                m_loss /= n
                c_loss /= n
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_p:{}, loss_d:{}, loss_m:{}, loss_c:{}'.format(epoch, i, p_loss, d_loss, m_loss, c_loss))
                p_loss = 0.
                d_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.set_mode()
        
    ###########################################################
    def imitating3(self, epoch):
        self.policy.set_mode(True)
        p_loss = 0.
        d_loss = 0.
        m_loss = 0.
        c_loss = 0.
        ##Nightop
        for i, data in enumerate(self.dataloader):
            [state, action, goal] = data
 
            state = state.float().to(DEVICE)
            action = action.float().to(DEVICE)
            goal = goal.float().to(DEVICE)        
            g = goal.multinomial(1)
            #g = deepcopy(goal[0]).tolist().index(1.)
            #print(goal)
            #print(g)
            s_a = torch.cat((state, action), 1)
            

            #train meta_policy#########################
            self.policy.meta_policy.zero_grad()
            action_1 = self.policy.meta_policy(state)
            a_1 = select_action(action_1, False).float()
            #print(action_1)
            #print(torch.LongTensor([g, ]).to(DEVICE))
            loss_m = nn.CrossEntropyLoss()(a_1, torch.LongTensor([g, ]).to(DEVICE))
            loss_m.backward()
            m_loss += loss_m.item()
            self.policy.meta_policy_optim.step()
            #train discriminator########################################
            '''
            self.policy.discriminator.zero_grad()
            #real
            predict = self.policy.discriminator(s_a).view(-1)
            label = torch.full((predict.size(0),), 1., device=DEVICE)  
            loss_r = self.criterion(predict, label)
            #fake
            a_weights = self.policy.policies[g](state)
            a = select_action(a_weights, False).float()
            predict = self.policy.discriminator(torch.cat((state, a), 1)).view(-1)
            label = torch.full((predict.size(0),), 0., device=DEVICE)
            loss_f = self.criterion(predict, label)
            #update
            loss_d = loss_r + loss_f
            d_loss += loss_d.item()
            loss_r.backward()
            loss_f.backward()
            self.policy.discriminator_optim.step()
            '''

            self.policy.network.zero_grad()
            #real
            predict = self.policy.network(s_a, 'dis').view(-1)
            label = torch.full((predict.size(0),), 1., device=DEVICE)  
            loss_r = self.criterion(predict, label)
            #fake
            a_weights = self.policy.policy(state, g)
            ##a_weights = self.policy.policies[g](state)
            a = select_action(a_weights, False).float()
            predict = self.policy.network(torch.cat((state, a), 1), 'dis').view(-1)
            predict_c = self.policy.network(torch.cat((state, a), 1), 'cla')
            label = torch.full((predict.size(0),), 0., device=DEVICE)
            loss_f = self.criterion(predict, label)
            loss_c = nn.CrossEntropyLoss()(predict_c, torch.LongTensor([g, ]).to(DEVICE))
            #update
            loss_d = loss_r + loss_f
            c_loss += loss_c.item()
            d_loss += loss_d.item()
            loss_r.backward(retain_graph=True)
            loss_f.backward(retain_graph=True)
            loss_c.backward(retain_graph=True)
            self.policy.network_optim.step()
            
            
            
            
            #train policy###################################
            '''
            self.policy.policies[g].zero_grad()
            
            a_weights = self.policy.policies[g](state)
            a = select_action(a_weights, False).float()
            #predict = self.policy.discriminator(torch.cat((state, a), 1)).view(-1)
            predict = self.policy.network(torch.cat((state, a), 1), 'dis').view(-1)
            predict_c = self.policy.network(torch.cat((state, a), 1), 'cla')
            label = torch.full((predict.size(0),), 1., device=DEVICE)
            loss_pd = self.criterion(predict, label)
            loss_pc = nn.CrossEntropyLoss()(predict_c, torch.LongTensor([g, ]).to(DEVICE))
            #update
            loss_p = loss_pd + loss_pc
            p_loss += loss_p.item()
            loss_pd.backward(retain_graph=True)
            loss_pc.backward(retain_graph=True)
            #loss_p.backward(retain_graph=True)
            self.policy.policies_optim[g].step()
            '''
            
            self.policy.policy.zero_grad()
            
            a_weights = self.policy.policy(state, g)
            a = select_action(a_weights, False).float()
            predict = self.policy.network(torch.cat((state, a), 1), 'dis').view(-1)
            predict_c = self.policy.network(torch.cat((state, a), 1), 'cla')
            label = torch.full((predict.size(0),), 1., device=DEVICE)
            loss_pd = self.criterion(predict, label)
            loss_pc = nn.CrossEntropyLoss()(predict_c, torch.LongTensor([g, ]).to(DEVICE))
            #update
            loss_p = loss_pd + loss_pc
            p_loss += loss_p.item()
            loss_pd.backward(retain_graph=True)
            loss_pc.backward(retain_graph=True)
            self.policy.policy_optim.step()
            
            #train classifier##################
            '''
            self.policy.policies[g].zero_grad()
            self.policy.classifier.zero_grad()
            
            output_c = self.policy.classifier(torch.cat((state, a), 1))
            #print(output_c)
            
            loss_c = nn.CrossEntropyLoss()(output_c, torch.LongTensor([g, ]).to(DEVICE))
            
            c_loss += loss_c.item()
            loss_c.backward()
            self.policy.policies_optim[g].step()
            self.policy.classifier_optim.step()
            '''


            n = int(self.num_minibatch/self.print_per_batch)
            if (i+1) % n == 0:
                p_loss /= n
                d_loss /= n
                m_loss /= n
                c_loss /= n
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_p:{}, loss_d:{}, loss_m:{}, loss_c:{}'.format(epoch, i, p_loss, d_loss, m_loss, c_loss))
                p_loss = 0.
                d_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.set_mode()


    def mle_loop(self, data):
        s, target_a = to_device(data)
        a = self.policy.policy(s)
        #for i in range(target_a.size(0)):
        #    target_a[i] = target_a[i] / torch.sum(target_a, 1)[i]
        #a = torch.softmax(a, 1)
        
        loss_a = self.multi_entropy_loss(a, target_a)
        return loss_a

    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            #loss_a = self.mle_loop(data)
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_train)
        logging.debug('<<dialog policy>> mle, epoch {}, loss_p:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best')
            
        return best
    '''
    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN
    
        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN
            
        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)
    '''
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.policy.meta_policy.state_dict(), directory + '/' + str(epoch) + '_mgail.meta.mdl')
        for i in range(7):
            torch.save(self.policy.policies[i].state_dict(), directory + '/' + str(epoch) + '_mgail.pol_'+str(i)+'.mdl')
        #torch.save(self.policy.discriminator.state_dict(), directory + '/' + str(epoch) + '_mgail.dis.mdl')
        #torch.save(self.policy.classifier.state_dict(), directory + '/' + str(epoch) + '_mgail.cla.mdl')
        #torch.save(self.policy.policy.state_dict(), directory + '/' + str(epoch) + '_mgail.pol.mdl')
        torch.save(self.policy.network.state_dict(), directory + '/' + str(epoch) + '_mgail.net.mdl')
        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
    



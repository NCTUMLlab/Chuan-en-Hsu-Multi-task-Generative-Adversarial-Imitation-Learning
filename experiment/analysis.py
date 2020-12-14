from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlu.milu.multiwoz import MILU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import numpy as np
import torch

import argparse
from convlab2.util.analysis_tool.analyzer import Analyzer
from convlab2.task.multiwoz.goal_generator2 import *

def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset_name", type=str, default="MultiWOZ", help="name of dataset")
    parser.add_argument("--sys_policy", type=str, default='RulePolicy', help="name of model")
    parser.add_argument("--sys_path", type=str, default='', help="path of model")
    parser.add_argument("--user", type=str, default='', help="path of model")
    parser.add_argument("--all", default=False, action='store_true', help="path of model")
    parser.add_argument("--skip", default=False, action='store_true', help="path of model")
    parser.add_argument("--num", type=int, default=100, help="path of model")
    #parser.add_argument("--log_path_suffix", type=str, default="", help="suffix of path of log file")
    #parser.add_argument("--log_dir_path", type=str, default="log", help="path of log directory")
    #parser.add_argument("--record", default=False, action='store_true', help="record trajectories")
    #parser.add_argument("--goal", type=str, default='', help="record goal")
    
    return parser.parse_args()

def get_policy(model_name, sys_path):
    sys_path = '/home/nightop/ConvLab-2/convlab2/policy/' + sys_path
    print('sys_policy sys_path:', sys_path)
    if model_name == "RulePolicy":
        from convlab2.policy.rule.multiwoz import RulePolicy
        policy_sys = RulePolicy()
    elif model_name == "PPO":
        from convlab2.policy.ppo import PPO
        if sys_path:
            policy_sys = PPO(False)
            policy_sys.load(sys_path)
        else:
            policy_sys = PPO.from_pretrained()
    elif model_name == "PG":
        from convlab2.policy.pg import PG
        if sys_path:
            policy_sys = PG(False)
            policy_sys.load(sys_path)
        else:
            policy_sys = PG.from_pretrained()
    elif model_name == "MLE":
        from convlab2.policy.mle.multiwoz import MLE
        if sys_path:
            policy_sys = MLE()
            policy_sys.load(sys_path)
        else:
            policy_sys = MLE.from_pretrained()
    elif model_name == "GDPL":
        from convlab2.policy.gdpl import GDPL
        if sys_path:
            policy_sys = GDPL(False)
            policy_sys.load(sys_path)
        else:
            policy_sys = GDPL.from_pretrained()
    elif model_name == "GAIL":
        from convlab2.policy.gail.multiwoz import GAIL
        if sys_path:
            policy_sys = GAIL()
            policy_sys.load(sys_path)   
    elif model_name == "MPPO":
        from convlab2.policy.mppo import MPPO
        if sys_path:
            policy_sys = MPPO()
            policy_sys.load(sys_path)
        else:
            policy_sys = MPPO.from_pretrained()
    elif model_name == 'MGAIL':
        from convlab2.policy.mgail.multiwoz import MGAIL
        if sys_path:
            policy_sys = MGAIL()
            policy_sys.load(sys_path)                
    return policy_sys

def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)
    
def set_user(user):
    # MILU
    user_nlu = BERTNLU()
    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character='usr')
    #'attraction', 'hotel', 'restaurant', 'train', 'taxi', 'hospital', 'police'
    if user:
        if user == '7':
            user_policy.policy.goal_generator = GoalGenerator_7()
        if user == 'attraction':
            user_policy.policy.goal_generator = GoalGenerator_attraction()
        elif user == 'hospital':
            user_policy.policy.goal_generator = GoalGenerator_hospital()
        elif user == 'hotel':
            user_policy.policy.goal_generator = GoalGenerator_hotel() 
        elif user == 'police':
            user_policy.policy.goal_generator = GoalGenerator_police()        
        elif user == 'restaurant':
            user_policy.policy.goal_generator = GoalGenerator_restaurant()
        elif user == 'taxi':
            user_policy.policy.goal_generator = GoalGenerator_taxi()
        elif user == 'train':
            user_policy.policy.goal_generator = GoalGenerator_train()
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    
    return user_nlu, user_dst, user_policy, user_nlg


def set_system(sys_policy, sys_path):
    # BERT nlu
    sys_nlu = BERTNLU()
    # simple rule DST
    sys_dst = RuleDST()
    # rule policy
    sys_policy = get_policy(sys_policy, sys_path)
    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    
    return sys_nlu, sys_dst, sys_policy, sys_nlg

if __name__ == '__main__':
    args = get_parser()

    set_seed(20200131)

    
    
    #set up user
    user_nlu, user_dst, user_policy, user_nlg = set_user(args.user)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
    
    #set up evaluater
    evaluator = MultiWozEvaluator()
    
    if not args.skip:
    
        sys_nlu, sys_dst, sys_policy, sys_nlg = set_system(args.sys_policy, args.sys_path)
        sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
        sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)
        
        sys_response = ''
        sess.init_session()
        print('init goal:')
        pprint(sess.evaluator.goal)
        print('-'*50)
        for i in range(20):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            print('user:', user_response)
            print('sys:', sys_response)
            print()
            if session_over is True:
                break
        print('task success:', sess.evaluator.task_success())
        print('book rate:', sess.evaluator.book_rate())
        print('inform precision/recall/f1:', sess.evaluator.inform_F1())
        print('-'*50)
        print('final goal:')
        pprint(sess.evaluator.goal)
        print('='*100)
    
    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')
    
    if not args.all:
        sys_nlu, sys_dst, sys_policy, sys_nlg = set_system(args.sys_policy, args.sys_path)
        sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
        
        analyzer.comprehensive_analyze_2(sys_agent=sys_agent, model_name='sys_agent', total_dialog=100) 
        #analyzer.compare_models([sys_agent, ], ['ppo_attraction_only', ], total_dialog=args.num)
    # if sys_nlu!=None, set use_nlu=True to collect more information
    
    else:
        agent_list = []
        agent_name = []
        for i in range(99, 100, 5):
            sys_nlu, sys_dst, sys_policy, sys_nlg = set_system(args.sys_policy, args.sys_path+str(i))
            sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')
            agent_list.append(sys_agent)
            agent_name.append(str(i))
            
        analyzer.compare_models(agent_list, agent_name, total_dialog=args.num)
        
        

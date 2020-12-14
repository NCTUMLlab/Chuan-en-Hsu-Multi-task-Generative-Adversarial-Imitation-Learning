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
import time
from datetime import datetime
from convlab2.task.multiwoz.goal_generator2 import *


def set_seed(r_seed, randomize=True):
    if not randomize:
        random.seed(r_seed)
        np.random.seed(r_seed)
        torch.manual_seed(r_seed)
    else:
        seed = int(time.time())
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

if __name__ == '__main__':


    # BERT nlu
    sys_nlu = BERTNLU()
    # simple rule DST
    sys_dst = RuleDST()
    # rule policy
    sys_policy = RulePolicy()
    from convlab2.policy.mgail.multiwoz import MGAIL
    policy_sys = MGAIL()
    policy_sys.load('/home/nightop/ConvLab-2/convlab2/policy/mgail/multiwoz/save/all/99')                
    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    # assemble
    sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')


    # MILU
    user_nlu = BERTNLU()
    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character='usr')
    #user_policy.policy.goal_generator = GoalGenerator_7()
    #user_policy.policy.goal_generator = GoalGenerator_restaurant()
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    # assemble
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
    
    
    
    evaluator = MultiWozEvaluator()
    sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)
    
    
    set_seed(20200131, True)

    sys_response = ''
    sess.init_session()
    print('init goal:')
    pprint(sess.evaluator.goal)
    print('-'*50)
    for i in range(20):
        sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
        print('user:', user_response)
        print('sys:', sys_response)
        if session_over is True:
            #print(sess.dialog_history)
            #print(evaluator.sys_da_array)
            #print(evaluator.usr_da_array)
            #print(evaluator.goal)
            #print(evaluator.cur_domain)
            #print(evaluator.booked)
            break
    print('task success:', sess.evaluator.task_success())
    print('book rate:', sess.evaluator.book_rate())
    print('inform precision/recall/f1:', sess.evaluator.inform_F1())
    print('-'*50)
    print('final goal:')
    pprint(sess.evaluator.goal)
    print('='*100)
    
    
    
    



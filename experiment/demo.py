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
    
    #1603813675
    set_seed(1603813675, False)

    sys_response = ''
    sess.init_session()
    print('init goal:')
    pprint(sess.evaluator.goal)
    print('-'*50)
    sess.demo(sys_response)


    
    
    
    


'''
seed:1603813675
{'hotel': {'info': {'internet': 'yes', 'parking': 'yes', 'stars': '3'},
           'reqt': {'pricerange': '?'}},
 'restaurant': {'info': {'name': 'the nirala'}, 'reqt': {'address': '?'}},
 'taxi': {'info': {'leaveAt': '16:15'},
          'reqt': {'car type': '?', 'phone': '?'}}}
--------------------------------------------------
user: I need a restaurant . I am looking for details on the the nirala restaurant .
system: They serve indian food .

user: I need a restaurant . I am looking for details on the the nirala restaurant .
sys: They serve indian food .
user: Can you give me the address ?
system: It is at 7 Milton Road Chesterton.

user: Can you give me the address ?
sys: It is at 7 Milton Road Chesterton.
user: I ' m looking for a hotel . I do n't want to have to pay for parking . I do need a place that has free wifi . It should have a 3 star rating .
system: Will gonville hotel be alright ? They are located at gonville place. I have 2 different options for you !.

user: I ' m looking for a hotel . I do n't want to have to pay for parking . I do need a place that has free wifi . It should have a 3 star rating .
sys: Will gonville hotel be alright ? They are located at gonville place. I have 2 different options for you !.
user: What is the price range of the hotel ?
system: Its listed as expensive .

user: What is the price range of the hotel ?
sys: Its listed as expensive .
user: I 'll need a ride there . Can you arrange a taxi for me ? Hi , I need to book a taxi , please ? I need to leave sometime after 16:15 .
system: A toyota is booked for you . You can reach them on 68029942094.

user: I 'll need a ride there . Can you arrange a taxi for me ? Hi , I need to book a taxi , please ? I need to leave sometime after 16:15 .
sys: A toyota is booked for you . You can reach them on 68029942094.
user: I need the car type please .
system: The model of the car was ford . Their contact number is 73561296129.

user: I need the car type please .
sys: The model of the car was ford . Their contact number is 73561296129.
user: Actually , I ' m all set . Thank you ! Bye !.
system: You are very welcome . Goodbye . You 're welcome . Have a good day !.

user: Actually , I ' m all set . Thank you ! Bye !.
sys: You are very welcome . Goodbye . You 're welcome . Have a good day !.
'''


'''
 'train': {'info': {'arriveBy': '16:15',
                    'day': 'sunday',
                    'departure': 'cambridge',
                    'destination': 'stansted airport'},
           'reqt': {'price': '?'}}}
'''
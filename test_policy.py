import os
from config_adv import Config
import torch
from torch.autograd import Variable 


def select_action(state, pro_policy, adv_policy, is_protagonist=True):

    state = torch.from_numpy(state).unsqueeze(0)

    if is_protagonist:
        action_mean, _, action_std = pro_policy(Variable(state))
    else:
        action_mean, _, action_std = adv_policy(Variable(state))

    action = torch.normal(action_mean, action_std)
    return action


def test_policy(env, pro_policy, adv_policy, running_state, filename):
    
    args = Config()
    
    checkpoint = torch.load(os.path.join(os.getcwd(), "checkpoints", filename))
    pro_policy.load_state_dict(checkpoint['state_dict_pro_policy'])
    adv_policy.load_state_dict(checkpoint['state_dict_adv_policy'])
    
    test_reward = 0
    
    t_action = temp_action()
    
    for _ in range(args.test_exp):
        state = env.reset()
        state = running_state(state)
        
        for _ in range(10000): # Don't infinite loop while learning

            pro_action = select_action(state, pro_policy, adv_policy, is_protagonist = True)
            pro_action = pro_action.data[0].numpy()

            adv_action = select_action(state, pro_policy, adv_policy, is_protagonist = False)
            adv_action = adv_action.data[0].numpy()*args.test_adv_fraction

            t_action.pro = pro_action
            t_action.adv = adv_action

            next_state, reward, done, _ = env.step(t_action)
            test_reward += reward

            next_state = running_state(next_state)
            
            if args.render:
                env.render()
            if done:
                break
            
            state = next_state
        
    test_reward /= args.test_exp


class temp_action(object):
    pro = None
    adv = None
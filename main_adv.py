#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:25:45 2019

@author: vineet
"""

from trpo_agent import TRPOAgent
from models import PolicyNetwork, ValueNetwork

import gym
from config import Config

def main():
    
    args = Config()
    env = gym.make(args.env_name)
    env.seed(args.seed)
    
    pro_policy = PolicyNetwork(num_inputs=env.pro.observation_space.shape[0], num_outputs=env.pro.action_space.shape[0])
    adv_policy = PolicyNetwork(num_inputs=env.adv.observation_space.shape[0], num_outputs=env.adv.action_space.shape[0])
    
    pro_value = ValueNetwork(num_inputs=env.pro.observation_space.shape[0])
    adv_value = ValueNetwork(num_inputs=env.adv.observation_space.shape[0])

    agent = TRPOAgent(env, pro_policy, adv_policy, pro_value, adv_value, is_protagonist = True)
    adv = TRPOAgent(env, pro_policy, adv_policy, pro_value, adv_value, is_protagonist = False)
    
    agent.step()
    adv.step()

if __name__ == '__main__':
    main()
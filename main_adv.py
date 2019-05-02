#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:25:45 2019

@author: vineet
"""

from trpo_agent_adv import TRPOAgent
from models import PolicyNetwork, ValueNetwork

from gym_env import GymEnv
from config_adv import Config
from itertools import count
from running_state import ZFilter

import logging
import os


def create_logger(filename):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(os.getcwd(), 'logs', filename+'.txt'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


#def main():

args = Config()
filename = 'adv_exp_' + '3'
logger = create_logger(filename)

env = GymEnv(env_name=args.env_name, seed = args.seed, adv_fraction=args.adv_fraction)
logger.info('{} Environment Created'.format(args.env_name))    

pro_policy = PolicyNetwork(num_inputs=env.observation_space.shape[0], num_outputs=env.pro_action_space.shape[0])
logger.info('Protagonist Policy Created')

adv_policy = PolicyNetwork(num_inputs=env.observation_space.shape[0], num_outputs=env.adv_action_space.shape[0])
logger.info('Adversary Policy Created')

pro_value = ValueNetwork(num_inputs=env.observation_space.shape[0])
logger.info('Value Function Network Created')
# adv_value = ValueNetwork(num_inputs=env.adv.observation_space.shape[0])

running_state = ZFilter((env.observation_space.shape[0],), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

agent = TRPOAgent(env, pro_policy, adv_policy, pro_value, running_state, running_reward, logger, filename, is_protagonist = True)
adv = TRPOAgent(env, pro_policy, adv_policy, pro_value, running_state, running_reward, logger, filename, is_protagonist = False)

for i_episode in count(1):    
    agent.step(i_episode=i_episode)
    adv.step(i_episode=i_episode)

#if __name__ == '__main__':
#    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:54:13 2019

@author: vineet
"""

class Config(object):
    """docstring for Config"""
    def __init__(self, env_name = "HalfCheetahAdv-v1"):
        super(Config, self).__init__()

        self.gamma = 0.995
        self.env_name = env_name
        self.tau = 0.97
        self.l2_reg = 1e-3
        self.max_kl = 1e-2
        self.damping = 1e-1
        self.seed = 543
        self.batch_size = 15000
        self.render = False
        self.log_interval = 1
        self.test_exp = 5
        self.adv_fraction = 1.0
        self.test_adv_fraction = 0.0
        
        self.n_pro_itr = 1
        self.n_adv_itr = 1
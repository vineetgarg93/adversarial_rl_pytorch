#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 12:10:23 2019

@author: vineet
"""

from trpo_agent import TRPOAgent

def main():

    agent = TRPOAgent()
    agent.step()

if __name__ == '__main__':
    main()
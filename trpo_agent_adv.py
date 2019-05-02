#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:26:40 2019

@author: vineet
"""

import torch
from torch.autograd import Variable
import scipy.optimize
import numpy as np
import os

from replay_memory_adv import Memory
from config import Config
from utils import set_flat_params_to, normal_log_density, get_flat_params_from, get_flat_grad_from

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


class TRPOAgent(object):
    """docstring for TRPOAgent"""
    def __init__(self, env, pro_policy, adv_policy, pro_value, running_state, running_reward, logger, filename, is_protagonist = True):
        super(TRPOAgent, self).__init__()

        self.args = Config()
        self.env = env
        self.logger = logger
        self.filename = filename + '_best.pth'
        
        self.best_reward = -np.inf
        
        torch.manual_seed(self.args.seed)
        
        self.pro_policy_model = pro_policy
        self.adv_policy_model = adv_policy        
        
        self.is_protagonist = is_protagonist

        if self.is_protagonist:
            self.policy_model = self.pro_policy_model
        else:
            self.policy_model = self.adv_policy_model

        self.value_model = pro_value
        
        self.running_state = running_state
        self.running_reward = running_reward

        self.temp_action = temp_action()

    
    def test_policy(self):
        
        
        for _ in range(self.args.test_exp):
            state = self.env.reset()
            state = self.running_state(state)
            
            for _ in range(10000): # Don't infinite loop while learning

                pro_action = self.select_action(state, is_protagonist = True)
                pro_action = pro_action.data[0].numpy()

                adv_action = self.select_action(state, is_protagonist = False)
                adv_action = adv_action.data[0].numpy()*self.args.test_adv_fraction

                temp_action.pro = pro_action
                temp_action.adv = adv_action

                next_state, reward, done, _ = self.env.step(temp_action)
                self.test_reward += reward

                next_state = self.running_state(next_state)
                
                if self.args.render:
                    self.env.render()
                if done:
                    break
                
                state = next_state
            
        self.test_reward /= self.args.test_exp
    
    
    def save_model(self, i_episode):
        
        self.logger.info('Performane Improved')
        self.logger.info('Episode {}\tTest Reward: {:.2f}'.format(
                        i_episode, self.test_reward))
        
        state = {
            'iteration': i_episode + 1,
            'state_dict_pro_policy': self.pro_policy_model.state_dict(),
            'state_dict_adv_policy': self.adv_policy_model.state_dict(),
            'state_dict_value_network': self.value_model.state_dict(),
            'best_reward':self.test_reward
        }
        
        self.logger.info("=>saving a new best checkpoint...")
        torch.save(state, os.path.join(os.getcwd(), 'checkpoints', self.filename))
    
    
    def step(self, i_episode):

        self.memory = Memory()
        self.reward_batch = 0
        self.num_episodes = 0
        self.test_reward = 0
        
        self.single_step()
        
        if not self.is_protagonist:
            self.test_policy()
            if self.test_reward > self.best_reward:
                self.save_model(i_episode)
                self.best_reward = self.test_reward

        if i_episode % self.args.log_interval == 0:
            
            if self.is_protagonist:
                self.logger.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                        i_episode, self.pro_reward_sum, self.reward_batch))
            else:
                self.logger.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tTest Reward: {:.2f}'.format(
                        i_episode, self.pro_reward_sum, self.reward_batch, self.test_reward))
#            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
#                    i_episode, self.pro_reward_sum, self.reward_batch))


    def single_step(self):

        num_steps = 0
        while num_steps < self.args.batch_size:
            state = self.env.reset()
            state = self.running_state(state)

            self.pro_reward_sum = 0
            self.adv_reward_sum = 0

            for t in range(10000): # Don't infinite loop while learning

                pro_action = self.select_action(state, is_protagonist = True)
                pro_action = pro_action.data[0].numpy()

                adv_action = self.select_action(state, is_protagonist = False)
                adv_action = adv_action.data[0].numpy()

                temp_action.pro = pro_action
                temp_action.adv = adv_action

                next_state, reward, done, _ = self.env.step(temp_action)
                
                pro_reward = reward
                adv_reward = -reward

                self.pro_reward_sum += pro_reward
                self.adv_reward_sum += adv_reward

                next_state = self.running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                self.memory.push(state, np.array([pro_action]), np.array([adv_action]), mask, next_state, pro_reward, adv_reward)

                if self.args.render:
                    self.env.render()
                if done:
                    break

                state = next_state
            num_steps += (t-1)
            self.num_episodes += 1
            self.reward_batch += self.pro_reward_sum

        self.reward_batch /= self.num_episodes
        batch = self.memory.sample()
        self.update_params(batch)


    def select_action(self, state, is_protagonist=True):

        state = torch.from_numpy(state).unsqueeze(0)

        if is_protagonist:
            action_mean, _, action_std = self.pro_policy_model(Variable(state))
        else:
            action_mean, _, action_std = self.adv_policy_model(Variable(state))

        action = torch.normal(action_mean, action_std)
        return action


    def update_params(self, batch):

        if self.is_protagonist:
            self.rewards = torch.Tensor(batch.pro_reward)
            self.actions = torch.Tensor(np.concatenate(batch.pro_action, 0))
        else:
            self.rewards = torch.Tensor(batch.adv_reward)
            self.actions = torch.Tensor(np.concatenate(batch.adv_action, 0))

        self.masks = torch.Tensor(batch.mask)
        self.states = torch.Tensor(batch.state)
        self.values = self.value_model(Variable(self.states))

        self.returns = torch.Tensor(self.actions.size(0),1)
        self.deltas = torch.Tensor(self.actions.size(0),1)
        self.advantages = torch.Tensor(self.actions.size(0),1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(self.rewards.size(0))):
            self.returns[i] = self.rewards[i] + self.args.gamma * prev_return * self.masks[i]
            self.deltas[i] = self.rewards[i] + self.args.gamma * prev_value * self.masks[i] - self.values.data[i]
            self.advantages[i] = self.deltas[i] + self.args.gamma * self.args.tau * prev_advantage * self.masks[i]

            prev_return = self.returns[i, 0]
            prev_value = self.values.data[i, 0]
            prev_advantage = self.advantages[i, 0]

        self.targets = Variable(self.returns)    
        
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(self.get_value_loss, 
                                                                get_flat_params_from(self.value_model).double().numpy(),
                                                                maxiter=25)

        set_flat_params_to(self.value_model, torch.Tensor(flat_params))

        self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()

        action_means, action_log_stds, action_stds = self.policy_model(Variable(self.states))
        self.fixed_log_prob = normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds).data.clone()

        self.trpo_step()


    def get_value_loss(self, flat_params):

        # states = args[0]
        # targets = args[1]

        set_flat_params_to(self.value_model, torch.Tensor(flat_params))
        for param in self.value_model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = self.value_model(Variable(self.states))

        value_loss = (values_ - self.targets).pow(2).mean()

        # weight decay
        for param in self.value_model.parameters():
            value_loss += param.pow(2).sum() * self.args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(self.value_model).data.double().numpy())


    def get_loss(self, volatile=False):

        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = self.policy_model(Variable(self.states))
        else:
            action_means, action_log_stds, action_stds = self.policy_model(Variable(self.states))
                
        log_prob = normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(self.advantages) * torch.exp(log_prob - Variable(self.fixed_log_prob))
        return action_loss.mean()


    def get_kl(self):

        mean1, log_std1, std1 = self.policy_model(Variable(self.states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


    def trpo_step(self):

        loss = self.get_loss()
        grads = torch.autograd.grad(loss, self.policy_model.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

        stepdir = self.conjugate_gradients(-loss_grad, 10)

        shs = 0.5 * (stepdir * self.Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / self.args.max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(self.policy_model)
        success, new_params = self.linesearch(prev_params, fullstep,
                                         neggdotstepdir / lm[0])
        set_flat_params_to(self.policy_model, new_params)

        return loss


    def Fvp(self, v):

        kl = self.get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.policy_model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, self.policy_model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * self.args.damping


    def linesearch(self,
                   x,
                   fullstep,
                   expected_improve_rate,
                   max_backtracks=10,
                   accept_ratio=.1):
        fval = self.get_loss(True).data
        print("fval before", fval.item())
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            set_flat_params_to(self.policy_model, xnew)
            newfval = self.get_loss(True).data
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                print("fval after", newfval.item())
                return True, xnew
        return False, x


    def conjugate_gradients(self, b, nsteps, residual_tol=1e-10):

        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.Fvp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x


class temp_action(object):
    pro = None
    adv = None
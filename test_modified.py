import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt
from itertools import count

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self._init_weight()
        

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.1)

        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
         

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

env_name = "HalfCheetah-v1"
env = gym.make(env_name)

num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]

#Hyper params:
hidden_size      = 256
lr               = 3e-4
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = -200

batch_size = 15000

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for i_episode in count(1):

    log_probs   = []
    values      = []
    states      = []
    next_states = []
    actions     = []
    rewards     = []
    masks       = []

    num_episodes = 0
    num_steps = 0
    reward_batch = 0

    while num_steps < batch_size:
        state = env.reset()

        reward_sum = 0
        for t in range(2000):

            state = torch.FloatTensor(state).to(device)
            state = torch.unsqueeze(state, 0)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.from_numpy(np.array(reward, dtype=np.float32)).unsqueeze(0).to(device))
            masks.append(torch.from_numpy(np.array(1 - done, dtype=np.float32)).unsqueeze(0).to(device))
            
            states.append(state)
            actions.append(action)

            reward_sum += reward

            if done:
                break
            
            state = next_state

        num_steps += (t - 1)
        num_episodes += 1
        reward_batch += reward_sum

    reward_batch /= num_episodes
    
    states    = torch.cat(states)
    actions   = torch.cat(actions)

    next_state = torch.FloatTensor(next_state).to(device)
    next_state = torch.unsqueeze(next_state, 0)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


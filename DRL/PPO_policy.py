import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import pandas as pd
import pytz
import math
from datetime import datetime
from torch.distributions import Categorical
import random

################################## PPO Policy ##################################

class RolloutBuffer: # Store and manage trajectories (rollouts)
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 256),
                        nn.Tanh(),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.Linear(256, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1)
                    )

    def forward(self): # Not implemented
        raise NotImplementedError

    def act(self, state): # Sample an action using the NN
        action_probs = self.actor(state) # Get action probabilities from the actor network
        dist = Categorical(action_probs) # Create a categorical distribution over the action probabilities
        action = dist.sample() # Sample an action from the distribution
        action_logprob = dist.log_prob(action) # Compute the log probability of the action
        return action.detach(), action_logprob.detach() # Detach the tensors before returning

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy #torch.squeeze removes extra size 1 dimensions

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.policy_old = ActorCritic(state_dim, action_dim).to(device) # A copy of the policy network, used to calculate the ratio of new and old action probabilities
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad(): # Don't need gradient, faster to run
            state = torch.FloatTensor(state).to(self.device) # Convert to type FloatTensor to be processed by NN
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob) # Update buffer with sampled state, action and logprob
        return action.item()

    def top_action(self, state): # Just used for testing -- returns most likely action
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy.actor(state)
        index = torch.argmax(probs)
        return index

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)): # Iterate through buffer in reverse order
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward) # Insert computed discounted reward to beginning of rewards list to maintain order

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device) # Convert to tensor type
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # Normalize

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device) # Convert into tensors
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy # was .01 for entropy
            #print(0.01*dist_entropy/loss)
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

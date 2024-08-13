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
np.random.seed(0)
#########################################################
# Parameters
L = 2 # Length of forecast horizon
LT = 2 # Lead time
epi_len = 256 # Length of one episode
ini_inv = [25,25] # Initial inventory levels
holding = [2,2] # Holding costs
lost_sales = [50,50] # Per unit lost sales costs
capacity = [60,60] # Inventory capacities
fixed_order = [10,10] # Fixed order costs per order
per_trans_item = 0 # Per unit cost for transshipment (either direction)
per_trans_order = 5 # Fixed cost per transshipment (either direction)
#########################################################

# Demand forecast function
# Note that at decision time t, demand for time t has already been realised
class RandomDemand:
    def __init__(self):
        self.list = np.random.uniform(0, 15, size=(2, epi_len + 3))
    
    def reset(self):
        self.list = np.random.uniform(0, 15, size=(2, epi_len + 3))
        
    def f(self, n, t): # Generate forecasts, f(n,t) corresponds to demand mean for retailer n at time t+1
        if n > 1:
            raise ValueError("Invalid retailer number")
        return self.list[n, t]

    # Function to generate demand history for the two retailers, of length epi_len+1
    def gen_demand(self):
        demand_hist_list = [] # List to hold demand histories for multiple retailers
        for k in range(2):
            demand_hist = []
            for i in range(1, epi_len + 2): # 1 extra demand generated so that last state has a next state
                random_demand = np.random.poisson(self.list[k, i])  # Poisson distribution with forecasted mean
                demand_hist.append(random_demand)
            demand_hist_list.append(demand_hist)
        return demand_hist_list

# def forec(n, t): # Demand forecast for retailer n=0,1 at time t=1,2,...
#     if n==0:
#       k=4*math.sin(math.pi*t/8)+6 # Not rounded, so non-integer
#     elif n==1:
#       k=4*math.cos(math.pi*t/8)+6
#     else:
#       raise ValueError("Invalid retailer number")
#     return k



# Define possible actions for inventory management
action_lists = [[0],[0, 5, 10, 15, 20],[0, 5, 10, 15, 20]]
action_map = [x for x in itertools.product(*action_lists)]

class Retailer:
    def __init__(self, demand_records, number, f): # Each retailer has its own number 0,1,2,... f is its forecast
        self.number = number # Retailer number
        self.inv_level = ini_inv[number]  # Initial inventory level
        self.holding_cost = holding[number]  # Holding cost per unit
        self.lost_sales_cost = lost_sales[number] # Cost per unit of lost sales
        self.pipeline = [0] * (LT-1) # List to track orders yet to arrive, of length (LT - 1)
        self.forecast = [f(number, t) for t in range(1,L+1)] # Forecast for time t+1
        self.capacity = capacity[number]  # Maximum inventory capacity
        self.demand_list = demand_records  # Historical demand records
        self.fixed_order_cost = fixed_order[number]  # Fixed cost for placing an order
        self.action = 0 # Order qty

    def reset(self, f):
        self.inv_level = ini_inv[self.number]
        self.pipeline = [0] * (LT-1)
        self.forecast = [f(self.number, t) for t in range(1,L+1)] # Forecast for time t+1

    def order_arrival(self, demand): # Get next state after pipeline inv arrives and demand is realized
        self.inv_level = min(self.capacity, self.inv_level + self.pipeline[0]) # Pipeline arrives, cannot exceed storage capacity
        self.inv_level -= demand
        # Update pipeline
        self.pipeline = np.concatenate((self.pipeline[1:],[self.action]))

class InvOptEnv:
    def __init__(self):
        self.rd = RandomDemand()
        self.demand_records = self.rd.gen_demand()
        self.n_retailers = 2
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(self.demand_records[i], i, self.rd.f))
        self.n_period = len(self.demand_records[0])
        self.current_period = 1
        self.state = np.array([retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in retailer.forecast] + \
                              [x for retailer in self.retailers for x in retailer.pipeline])

    def reset(self): # Resets state of all retailers and DCs by calling their respective reset methods
        self.rd.reset()
        self.demand_records = self.rd.gen_demand() # New instance of demand realizations
        for retailer in self.retailers:
            retailer.reset(self.rd.f)
        self.current_period = 1
        self.state = np.array([retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in retailer.forecast] + \
                              [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state

    def step(self, action):
        action_modified = action_map[action]
        trans = action_modified[0] # Transshipment quantity, possibly infeasible
        # Make transshipment quantity feasible
        if trans > 0 and self.retailers[0].inv_level < trans:
            trans = 0
        elif trans < 0 and self.retailers[1].inv_level < -trans:
            trans = 0
        trans_cost = trans * per_trans_item + (trans!=0) * per_trans_order # Transshipment cost

        hl_cost_total = 0
        order_cost = 0
        # Calculate sum of order, holding, lost sales costs
        for i, retailer in enumerate(self.retailers): # Iterate through retailers
            retailer.action = action_modified[i+1] # Qty ordered by retailer
            # Get order costs
            order_cost += (retailer.action>0) * retailer.fixed_order_cost
            # Do transshipment
            if retailer.number==0:
              retailer.inv_level -= trans
            else:
              retailer.inv_level += trans
            # Get holding/lost sales cost
            if retailer.inv_level < 0: # Get lost sales costs and set to zero
              hl_cost_total += - retailer.inv_level * retailer.lost_sales_cost
              retailer.inv_level = 0
            else:
              hl_cost_total += retailer.inv_level * retailer.holding_cost
        reward = - trans_cost - hl_cost_total - order_cost

        self.current_period += 1
        if self.current_period >= self.n_period:
            terminate = True
        else:
            terminate = False
        # Update forecasts
        for i,retailer in enumerate(self.retailers):
            retailer.forecast = [self.rd.f(i,k) for k in range(self.current_period, self.current_period+L)] # No +1
        # Update inv levels and pipelines
        for retailer, demand in zip(self.retailers, self.demand_records):
            retailer.order_arrival(demand[self.current_period - 2]) # -2 not -1
        self.state = np.array([retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in retailer.forecast] + \
                              [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state, reward, terminate

# PPO Implementation using PyTorch
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if False: #(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


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
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
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
            state = torch.FloatTensor(state).to(device) # Convert to type FloatTensor to be processed by NN
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob) # Update buffer with sampled state, action and logprob
        return action.item()

    def top_action(self, state): # Just used for testing -- returns most likely action
        state = torch.FloatTensor(state).to(device)
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

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device) # Convert to tensor type
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # Normalize

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device) # Convert into tensors
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

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

################################### Training ###################################
def train():
    print("============================================================================================")


    max_ep_len = epi_len                   # max timesteps in one episode
    max_training_timesteps = int(max_ep_len * 40000)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    plot_freq = max_ep_len * 100      # plots
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len       # update policy every n timesteps, was /2
    K_epochs = 20               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.00005       # learning rate for actor network
    lr_critic = 0.0001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    state_dim = 2 * (1 + L + LT - 1)
    action_dim = len(action_map)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = datetime.now(pytz.timezone('Asia/Singapore')).replace(microsecond=0)
    print("Started training at (SGT) : ", start_time)

    print("============================================================================================")


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 1
    cost_list = []
    best_so_far = 10**10
    
    # training loop
    if True:
        # Generate forecasts and demand
        env = InvOptEnv()
        while time_step <= max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for _ in range(1, max_ep_len+1):

                # select action with policy

                action = ppo_agent.select_action(state)
                state, reward, done = env.step(action)
            
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # printing average reward
                if time_step % print_freq == 0:
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
                    cost_list.append(-print_avg_reward)
                    print("Episode : {} \t\t Timestep : {} \t\t Average Cost : {}".format(i_episode, time_step, -print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0
                    if -print_avg_reward < best_so_far:
                        print("Best so far")
                        best_so_far = -print_avg_reward
                        torch.save(ppo_agent.policy.state_dict(), 'policy_best_so_far.pt')

                # plot graphs and write to csv
                if time_step % plot_freq == 0:
                    #print('Plot', time_step)
                    #print(ppo_agent.buffer.states)
                    #print(len(ppo_agent.buffer.states))
                    plotdone = False
                    while not plotdone:
                        try:
                            plt.figure(figsize=(15, 10))
                            plt.subplot(131) # States
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(x[0]) for x in ppo_agent.buffer.states])
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(x[1]) for x in ppo_agent.buffer.states])
                            plt.subplot(132) # Actions
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(action_map[x][1]) for x in ppo_agent.buffer.actions])
                            plt.plot(np.arange(1,max_ep_len+1,1),[int(action_map[x][2]) for x in ppo_agent.buffer.actions])
                            plt.subplot(133) # Rewards
                            plt.plot(np.arange(1,max_ep_len+1,1),[-int(x) for x in ppo_agent.buffer.rewards])
                            plt.savefig('plot.png')
                            plt.close()

                            # Plot graph of costs by episode
                            plt.plot(10*np.arange(1,len(cost_list)+1,1),cost_list)
                            plt.savefig('cost.png')
                            plt.close()
                            #print([a.cpu() for a in ppo_agent.buffer.states])
                            #print([int(action_map[x][0]) for x in ppo_agent.buffer.actions])
                            # write to csv
                            df = pd.DataFrame({
                            'states': [a.cpu() for a in ppo_agent.buffer.states],
                            'transship': [int(action_map[x][0]) for x in ppo_agent.buffer.actions],
                            'actions1': [int(action_map[x][1]) for x in ppo_agent.buffer.actions],
                            'actions2': [int(action_map[x][2]) for x in ppo_agent.buffer.actions],
                            'logprobs': [a.cpu() for a in ppo_agent.buffer.logprobs],
                            'rewards': ppo_agent.buffer.rewards,
                            'is_terminals': ppo_agent.buffer.is_terminals
                            })
                            df.to_csv('trajectory_non_trans.csv', index=False)
                            plotdone = True
                            
                            torch.save(ppo_agent.policy.state_dict(), 'policy_non_trans.pt')

                        except ValueError:
                            continue
                # update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()
                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

if __name__ == '__main__':
    train()
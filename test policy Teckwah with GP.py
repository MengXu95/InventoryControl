import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
import pandas as pd
import math
from statistics import mean, stdev
from datetime import datetime
from torch.distributions import Categorical
import MTGP_niching.LoadIndividual as mtload
from MTGP_niching.replenishment import *
from MTGP_niching.transshipment import *

np.random.seed(0)
#########################################################
# Parameters
L = 2 # Length of forecast horizon
LT = 2 # Lead time
epi_len = 64 # Length of one episode
ini_inv = [10000,0] # Initial inventory levels
holding = [2,0] # Holding costs
lost_sales = [50,0] # Per unit lost sales costs
capacity = [200000,0] # Inventory capacities
fixed_order = [10000,0] # Fixed order costs per order
per_trans_item = 0 # Per unit cost for transshipment (either direction)
per_trans_order = 5 # Fixed cost per transshipment (either direction)
#########################################################

# Obtain testing demand data from csv file
test_demand = pd.read_csv('teckwah.csv')
demand_test = []
for k in range(10):
    demand_hist_list = test_demand.iloc[2*k : 2*k +2, 1:].to_numpy()
    demand_test.append(demand_hist_list)


# Define possible actions for inventory management
action_lists = [[0],[0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000],[0]]
action_map = [x for x in itertools.product(*action_lists)]

class Retailer:
    def __init__(self, demand_records, number): # Each retailer has its own number 0,1,2,...
        self.number = number # Retailer number
        self.inv_level = ini_inv[number]  # Initial inventory level
        self.order_quantity_limit = 20  # Maximum order quantity
        self.holding_cost = holding[number]  # Holding cost per unit
        self.lost_sales_cost = lost_sales[number] # Cost per unit of lost sales
        self.pipeline = [0] * (LT-1) # List to track orders yet to arrive, of length (LT - 1)
        self.forecast = [demand_records[number, t] for t in range(1,L+1)] # Forecast for time t+1
        self.capacity = capacity[number]  # Maximum inventory capacity
        self.demand_list = demand_records  # Historical demand records
        self.fixed_order_cost = fixed_order[number]  # Fixed cost for placing an order
        self.action = 0 # Order qty

    def reset(self):
        self.inv_level = ini_inv[self.number]
        self.pipeline = [0] * (LT-1)
        #self.forecast = [demand_records[number, t] for t in range(1,L+1)] # Forecast for time t+1

    def order_arrival(self, demand): # Get next state after pipeline inv arrives and demand is realized
        self.inv_level = min(self.capacity, self.inv_level + self.pipeline[0]) # Pipeline arrives, cannot exceed storage capacity
        self.inv_level -= demand
        # Update pipeline
        self.pipeline = np.concatenate((self.pipeline[1:],[self.action]))

class MultiEchelonInvOptEnv:
    def __init__(self, demand_records):
        self.n_retailers = 2
        self.retailers = []
        for i in range(self.n_retailers):
            self.retailers.append(Retailer(demand_records[i], i))
        self.n_period = len(demand_records[0])
        self.current_period = 1
        self.state = np.array([retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in retailer.forecast] + \
                              [x for retailer in self.retailers for x in retailer.pipeline])
        self.variable_order_cost = 10
        self.demand_records = demand_records

    def reset(self): # Resets state of all retailers and DCs by calling their respective reset methods
        for retailer in self.retailers:
            retailer.reset()
        self.current_period = 1
        self.state = np.array([retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in retailer.forecast] + \
                              [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state

    def step(self, action, no_trans = False): # no_trans forces transshipment=0 if True
        action_modified = action_map[action]
        trans = action_modified[0] # Transshipment quantity, possibly infeasible
        # Make transshipment quantity feasible
        if trans > 0 and self.retailers[0].inv_level < trans:
            trans = 0
        elif trans < 0 and self.retailers[1].inv_level < -trans:
            trans = 0
            
        if no_trans: # Force no transshipment
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
        #for i,retailer in enumerate(self.retailers):
            #retailer.forecast = [forec(i,k) for k in range(self.current_period, self.current_period+L)] # No +1
        # Update inv levels and pipelines
        for retailer, demand in zip(self.retailers, self.demand_records):
            retailer.order_arrival(demand[self.current_period - 2]) # -2 not -1
        self.state = np.array([retailer.inv_level for retailer in self.retailers] + [x for retailer in self.retailers for x in retailer.forecast] + \
                              [x for retailer in self.retailers for x in retailer.pipeline])
        return self.state, reward, terminate

# PPO Implementation using PyTorch
################################## set device ##################################
#print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
#if(torch.cuda.is_available()):
#    device = torch.device('cuda:0')
#    torch.cuda.empty_cache()
#    print("Device set to : " + str(torch.cuda.get_device_name(device)))
#else:
#    print("Device set to : cpu")
#print("============================================================================================")


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
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

################ PPO hyperparameters ################
update_timestep = epi_len       # update policy every n timesteps, was /2
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

################# Testing procedure ################
# Load PPO policy
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
ppo_agent.policy.load_state_dict(torch.load('policy teckwah.pt'))
#ppo_agent.policy_old.load_state_dict(torch.load('policy_best_so_far_no_trans.pt'))

##############################################################################################################
# Testing the policies-- for DRL, choose action based on highest probs

# For sS policy
# Implement sS action, assume LT=2
def sS_action(state, sS1, sS2):
    if state[0] + state[-2] < sS1[0]:
        order1 = sS1[1] - state[0] - state[-2]
    else:
        order1 = 0
    if state[1] + state[-1] < sS2[0]:
        order2 = sS2[1] - state[1] - state[-1]
    else:
        order2 = 0
    return order1, order2

# For DRL policy, with or without transshipment
def DRL_action(state, no_trans = False): # Returns the DRL argmax action. no_trans sets no transshipment
    index = ppo_agent.top_action(state)
    action = action_map[index]
    trans = action[0]
    if no_trans:
        trans = 0
    elif trans > state[0] and trans > 0:
        trans = 0
    elif -trans > state[1] and trans < 0:
        trans = 0
    return trans, action[1], action[2]        

# Get next state and reward given feasible action that has no transshipment
def any_step(state, action, current_period, demand, trans = 0): # Action is a 2-vector of order quantities
    if trans != 0:
        if trans > 0 and state[0] < trans:
            trans = 0
        elif trans < 0 and state[1] < -trans:
            trans=0
            
    trans_cost = trans * per_trans_item + (trans!=0) * per_trans_order # Transshipment cost
    
    hl_cost_total = 0
    order_cost = 0
    new_state = [state[0], state[1]] # To be calculated
    # Calculate sum of order, holding, lost sales costs
    for i in range(2): # Iterate through retailers
        order_qty = action[i] # 2-vector containing order qtys
        # Get order costs
        order_cost += (order_qty>0) * fixed_order[i]
        # Do transshipment
        if i == 0:
            new_state[i] -= trans
        else:
            new_state[i] += trans
        # Get holding/lost sales cost
        if new_state[i] < 0: # Get lost sales costs and set to zero
            hl_cost_total += - new_state[i] * lost_sales[i]
            new_state[i] = 0
        else: # Get holding cost
            hl_cost_total += new_state[i] * holding[i]
    reward = - trans_cost - hl_cost_total - order_cost

    # Update forecasts
    forecast1 = [demand[0,k] for k in range(current_period, current_period+L)] 
    forecast2 = [demand[1,k] for k in range(current_period, current_period+L)] 

    # Update inv levels and pipelines - assumes LT=2
    pipeline1 = state[-2*(LT-1):-(LT-1)]
    pipeline2 = state[-(LT-1):]
    new_state[0] = min(capacity[0], new_state[0] + pipeline1[0]) # Order arrival, subject to capacity
    new_state[1] = min(capacity[1], new_state[1] + pipeline2[0])
    new_state[0] -= demand[0][current_period-1] # Demand realization
    new_state[1] -= demand[1][current_period-1]
    
    pipeline1 = [action[0]]
    pipeline2 = [action[1]]
    #print(pipeline1)
    #print(pipeline2)
    next_state = np.array(new_state + forecast1 + forecast2 + pipeline1 + pipeline2)
    #print(next_state)
    action_taken = [trans, action[0], action[1]]
    return next_state, action_taken, reward

#########################################################
# Begin testing
# For GP policy
# read GP policy
print('\nBegin testing GP for policy from each generation: ')
randomSeeds = 0
dataSetName = 'Teckwah'
dict_best_individuals = mtload.load_individual_from_gen(randomSeeds, dataSetName)
# training_time = mtload.load_training_time(randomSeeds, dataSetName)
# min_fitness = mtload.load_min_fitness(randomSeeds, dataSetName)
# print('\nTraining time: ')
# print(training_time)
# print('Training min_fitness: ')
# print(min_fitness)
# print('\n')

best_invlvls_GP = [np.Infinity] # List of average inventory levels
best_fill_GP = [np.Infinity]
best_cost_GP = [np.Infinity] # List of rewards for each demand realization (DRL no transshipment scenario)
best_cost_GP_all_gens = []
for idx in range(len(dict_best_individuals)):
    algo = 'gen_' + str(idx) + '_MTGP_test'
    individual = dict_best_individuals.get(idx)
    if len(individual) == 1:
        site1_candidate = action_lists[1]
        replenishment_site1 = individual[0]
    else:
        site1_candidate = action_lists[1]
        site2_candidate = action_lists[2]
        replenishment_site1 = individual[0]
        replenishment_site2 = individual[1]
    invlvls_GP = [] # List of average inventory levels
    fill_GP = []
    cost_GP = [] # List of rewards for each demand realization (DRL no transshipment scenario)
    for demand in demand_test:
        GP_states = []
        GP_actions = []
        GP_rewards = []
        reward_total = 0 # Running total reward for current demand realization
        state = [ini_inv[0],ini_inv[1],demand[0,0],demand[0,1],demand[1,0],demand[1,1],0,0] # Initial state
        current_period = 1
        while current_period <= epi_len:
            quantity_site1 = GP_pair_S_test(state, replenishment_site1)

            index_site1 = 0
            min_dis = np.Infinity
            for i in range(len(site1_candidate)):
                dis = np.abs(quantity_site1 - site1_candidate[i])
                if dis < min_dis:
                    index_site1 = i
                    min_dis = dis

            action = [0, site1_candidate[index_site1], 0]

            trans = action[0]
            GP_states.append(state)
            state, action, reward = any_step(state, action[1:], current_period, demand, trans = trans) # Remove first element (transshipment) of action
            GP_actions.append(action)
            reward_total += -reward
            GP_rewards.append(reward)
            current_period += 1
        cost_GP.append(reward_total)
        invlvls_GP.append(mean([x[0]+x[1] for x in GP_states]))
        fill_GP.append(mean([int(x[0]>=0) + int(x[1]>=0) for x in GP_states]))
    if mean(cost_GP) < mean(best_cost_GP):
        best_cost_GP = cost_GP
    if mean(invlvls_GP) < mean(best_invlvls_GP):
        best_invlvls_GP = invlvls_GP
    if mean(fill_GP) < mean(best_fill_GP):
        best_fill_GP = fill_GP
    print(cost_GP)
    print(mean(cost_GP))
    best_cost_GP_all_gens.append(mean(cost_GP))
    df = pd.DataFrame({
                        'states': [torch.Tensor(x) for x in GP_states],
                        'transshipment': [x[0] for x in GP_actions],
                        'actions1': [x[1] for x in GP_actions],
                        'actions2': [x[2] for x in GP_actions],
                        'rewards': GP_rewards,
                        })
    df.to_csv('./GP_trajectory/GP_trajectory_gen_' + str(idx) + '.csv', index=False)

#########painting curve###########
# Plotting the values
plt.plot(best_cost_GP_all_gens, marker='o', linestyle='-', color='b')

# Adding titles and labels
plt.title('Convergence of GP')
plt.xlabel('Generation')
plt.ylabel('Cost')
# Save the plot to a PDF file
plt.savefig('convergence_of_GP.pdf')
# Display the plot
plt.show()
#########painting curve###########



#########################################################
# Begin testing
# For DRL policy
print('\nBegin testing DRL: ')
invlvls_DRL = [] # List of average inventory levels
fill_DRL = []
cost_DRL = [] # List of rewards for each demand realization (DRL no transshipment scenario)
for demand in demand_test:
    DRL_states = []
    DRL_actions = []
    DRL_rewards = []
    reward_total = 0 # Running total reward for current demand realization
    state = [ini_inv[0],ini_inv[1],demand[0,0],demand[0,1],demand[1,0],demand[1,1],0,0] # Initial state
    current_period = 1
    while current_period <= epi_len:
        action = DRL_action(state, no_trans = False) # May not be feasible, wait till any_step to get feasible action
        trans = action[0]
        DRL_states.append(state)
        state, action, reward = any_step(state, action[1:], current_period, demand, trans = trans) # Remove first element (transshipment) of action
        DRL_actions.append(action)
        reward_total += -reward
        DRL_rewards.append(reward)
        current_period += 1
    cost_DRL.append(reward_total)
    invlvls_DRL.append(mean([x[0]+x[1] for x in DRL_states]))
    fill_DRL.append(mean([int(x[0]>=0) + int(x[1]>=0) for x in DRL_states]))
print(cost_DRL)
print(mean(cost_DRL))
df = pd.DataFrame({
                    'states': [torch.Tensor(x) for x in DRL_states],
                    'transshipment': [x[0] for x in DRL_actions],
                    'actions1': [x[1] for x in DRL_actions],
                    'actions2': [x[2] for x in DRL_actions],
                    'rewards': DRL_rewards,
                    })
df.to_csv('trajectory.csv', index=False)


print('\nBegin testing sS: ')
sS_params = [[21000,29000],[20,24]] # Optimal sS params
sS1, sS2 = sS_params
cost_sS = [] # List of rewards for each demand realization (sS scenario)
invlvls_sS = []
fill_sS = []
for demand in demand_test:
    sS_states = []
    sS_actions = []
    sS_rewards = []
    reward_total = 0 # Running total reward for current demand realization
    state = [ini_inv[0],ini_inv[1],demand[0,0],demand[0,1],demand[1,0],demand[1,1],0,0] # Initial state
    current_period = 1
    while current_period <= epi_len:
        action = sS_action(state, sS1, sS2)
        sS_states.append(state)
        sS_actions.append(action)
        state, _, reward = any_step(state, action, current_period, demand)
        reward_total += -reward
        sS_rewards.append(reward)
        current_period += 1
    cost_sS.append(reward_total)
    invlvls_sS.append(mean([x[0]+x[1] for x in sS_states]))
    fill_sS.append(mean([int(x[0]>=0) + int(x[1]>=0) for x in sS_states]))
print(cost_sS)
print(mean(cost_sS))
df = pd.DataFrame({
                    'states': [torch.Tensor(x) for x in sS_states],
                    'actions1': [x[0] for x in sS_actions],
                    'actions2': [x[1] for x in sS_actions],
                    'rewards': sS_rewards,
                    })
df.to_csv('trajectory_sS.csv', index=False)


print('\nFinal comparison: ')
print([mean(best_cost_GP), mean(cost_DRL), mean(cost_sS)])
print([stdev(best_cost_GP), stdev(cost_DRL), stdev(cost_sS)])

print('\nDiff between GP and DRL: ')
DRL_diff = [y-x for x,y in zip(best_cost_GP, cost_DRL)]
print(mean(DRL_diff))

print('\nDiff between GP and sS: ')
sS_diff = [y-x for x,y in zip(best_cost_GP, cost_sS)]
print(mean(sS_diff))

print('\nDiff between DRL and sS: ')
sS_diff = [y-x for x,y in zip(cost_DRL, cost_sS)]
print(mean(sS_diff))

print('\nFinal inventory level and fill: ')
print([mean(invlvls_GP), mean(invlvls_DRL), mean(invlvls_sS), mean(fill_GP), mean(fill_DRL), mean(fill_sS)])
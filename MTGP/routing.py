import math

import simpy
import random
import numpy as np
import torch

'''
this module contains the machine routing rules used for comparison
routing agents may choose to follow one of following rules
or choose to use trained parameters for decision-making
'''

# Benchmark, as the worst possible case
def random_routing(idx, data, job_pt, job_slack, wc_idx, *args):
    machine_idx = np.random.randint(len(job_pt))
    return machine_idx

def TT(idx, data, job_pt, job_slack, wc_idx, *args): # shortest total waiting time
    # axis=0 means choose along columns
    # print("routing data:", data)
    rank = np.argmin(data, axis=0)
    machine_idx = rank[0]
    return machine_idx

def ET(idx, data, job_pt, job_slack, wc_idx, *args): # minimum exceution time
    machine_idx = np.argmin(job_pt)
    return machine_idx

def EA(idx, data, job_pt, job_slack, wc_idx, *args): # earliest available
    #print(data, np.transpose(data))
    rank = np.argmin(data, axis=0)
    machine_idx = rank[1]
    return machine_idx

def SQ(idx, data, job_pt, job_slack, wc_idx, *args): # shortest queue
    rank = np.argmin(data, axis=0)
    machine_idx = rank[2]
    return machine_idx

def CT(idx, data, job_pt, job_slack, wc_idx, *args): # earliest completion time
    #print(data,job_pt)
    completion_time = np.array(data)[:,1].clip(0) + np.array(job_pt)
    machine_idx = completion_time.argmin()
    return machine_idx

def UT(idx, data, job_pt, job_slack, wc_idx, *args): # lowest utilization rate
    rank = np.argmin(data, axis=0)
    machine_idx = rank[3]
    return machine_idx

def GP_R1(idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming
    data = np.transpose(data)
    sec1 = min(2 * data[2] * np.max([data[2]*job_pt/data[1] , job_pt*data[0]*data[0]], axis=0))
    sec2 = data[2] * job_pt - data[1]
    sum = sec1 + sec2
    machine_idx = sum.argmin()
    return machine_idx

def GP_R2(idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming
    data = np.transpose(data) #todo: check what's the data here!!!
    sec1 = data[2]*data[2], (data[2]+job_pt)*data[2]
    sec2 = np.min([data[1],args[0]/(data[1]*args[0]-1)],axis=0)
    sec3 = -data[2] * args[0]
    sec4 = data[2] * job_pt * np.max([data[0], np.min([data[1],job_pt],axis=0)/(args[0])],axis=0)
    sec5 = np.max([data[2]*data[2], np.ones_like(data[2])*(args[1]-args[0]-1), (data[2]+job_pt)*np.min([data[2],np.ones_like(data[2])*args[1]],axis=0)],axis=0)
    sum = sec1 - sec2 * np.max([sec3+sec4/sec5],axis=0)
    machine_idx = sum.argmin()
    return machine_idx



def GP_pair_R_test(state, tree_R): # genetic programming rule 1
    individualvalue = treeNode_R_test(tree_R, 0, state)  # todo: actually, this should be used for sequencing rule
    return individualvalue
    # if isinstance(individualvalue, (np.int64, np.float64, float, int)):
    #     return 0  # todo: need to check if this is right!!! by mengxu 2022.10.15
    # transshipment = individualvalue.argmin()
    # return transshipment


def treeNode_R_test(tree, index, data):
    if tree[index] == 'add':
        return treeNode_R_test(tree, index + 1, data) + treeNode_R_test(tree, index + 2, data)
    elif tree[index] == 'subtract':
        return treeNode_R_test(tree, index + 1, data) - treeNode_R_test(tree, index + 2, data)
    elif tree[index] == 'multiply':
        return treeNode_R_test(tree, index + 1, data) * treeNode_R_test(tree, index + 2, data)
    elif tree[index] == 'protected_div':
        return protected_div(treeNode_R_test(tree, index + 1, data), treeNode_R_test(tree, index + 2, data))
    elif tree[index] == 'maximum':
        return np.maximum(treeNode_R_test(tree, index + 1, data), treeNode_R_test(tree, index + 2, data))
    elif tree[index] == 'minimum':
        return np.minimum(treeNode_R_test(tree, index + 1, data), treeNode_R_test(tree, index + 2, data))
    elif tree[index] == 'lf':  # add by mengxu 2022.11.08
        ref = treeNode_R_test(tree, index + 1, data)
        if isinstance(ref, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-ref))
        else:
            for i in range(len(ref)):
                ref[i] = 1 / (1 + np.exp(-ref[i]))
            return ref
    elif tree[index] == 'INL1':
        return data[0]
    elif tree[index] == 'INL2':
        return data[1]
    elif tree[index] == 'FC11':
        return data[2]
    elif tree[index] == 'FC12':
        return data[3]
    elif tree[index] == 'FC21':
        return data[4]
    elif tree[index] == 'FC22':
        return data[5]
    elif tree[index] == 'PIP1':
        return data[6]
    elif tree[index] == 'PIP2':
        return data[7]


def GP_evolve_R(data, tree_R): # genetic programming evolved sequencing rule
    individualvalue = treeNode_R(tree_R, 0, data)  # todo: actually, this should be used for sequencing rule
    return individualvalue
    # if isinstance(individualvalue, (np.int64, np.float64, float, int)):
    #     return 0  # todo: need to check if this is right!!! by mengxu 2022.10.15
    # inventory_replenishment = individualvalue.argmin()
    # return inventory_replenishment


def treeNode_R(tree, index, data):
    if tree[index].arity == 2:
        if tree[index].name == 'add':
            return treeNode_R(tree, index + 1, data) + treeNode_R(tree, index + 2, data)
        elif tree[index].name == 'subtract':
            return treeNode_R(tree, index + 1, data) - treeNode_R(tree, index + 2, data)
        elif tree[index].name == 'multiply':
            return treeNode_R(tree, index + 1, data) * treeNode_R(tree, index + 2, data)
        elif tree[index].name == 'protected_div':
            return protected_div(treeNode_R(tree, index + 1, data), treeNode_R(tree, index + 2, data))
        elif tree[index].name == 'maximum':
            return np.maximum(treeNode_R(tree, index + 1, data), treeNode_R(tree, index + 2, data))
        elif tree[index].name == 'minimum':
            return np.minimum(treeNode_R(tree, index + 1, data), treeNode_R(tree, index + 2, data))
    elif tree[index].arity == 1:
        if tree[index].name == 'lf':  # add by mengxu 2022.11.08
            ref = treeNode_R(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref))
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref
    elif tree[index].arity == 0:
        if tree[index].name == 'INL1':
            return data[0]
        elif tree[index].name == 'INL2':
            return data[1]
        elif tree[index].name == 'FC11':
            return data[2]
        elif tree[index].name == 'FC12':
            return data[3]
        elif tree[index].name == 'FC21':
            return data[4]
        elif tree[index].name == 'FC22':
            return data[5]
        elif tree[index].name == 'PIP1':
            return data[6]
        elif tree[index].name == 'PIP2':
            return data[7]




def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x
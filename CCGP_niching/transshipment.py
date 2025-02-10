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

def GP_pair_R_test(state, tree_R): # genetic programming rule 1
    transshipment,length_tree = treeNode_R_test(tree_R, 0, state)  # todo: actually, this should be used for sequencing rule
    return transshipment


def treeNode_R_test(tree, index, data):
    if tree[index] == 'add':
        left, length_left = treeNode_R_test(tree, index + 1, data)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data)
        return safe_add(left, right), length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_R_test(tree, index + 1, data)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data)
        return safe_subtract(left, right), length_left+length_right+1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_R_test(tree, index + 1, data)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data)
        return safe_multiply(left, right), length_left+length_right+1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_R_test(tree, index + 1, data)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data)
        return protected_div(left, right), length_left+length_right+1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_R_test(tree, index + 1, data)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data)
        return np.maximum(left, right), length_left+length_right+1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_R_test(tree, index + 1, data)
        right, length_right = treeNode_R_test(tree, index + length_left + 1, data)
        return np.minimum(left, right), length_left+length_right+1
    elif tree[index] == 'protected_sqrt':
        child, length_child = treeNode_R_test(tree, index + 1, data)
        return protected_sqrt(child),length_child+1
    elif tree[index] == 'square':
        child, length_child = treeNode_R_test(tree, index + 1, data)
        return safe_square(child),length_child+1
    elif tree[index] == 'lf':  # add by mengxu 2022.11.08
        child, length_child = treeNode_R_test(tree, index + 1, data)
        if isinstance(child, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-child)),length_child+1
        else:
            for i in range(len(child)):
                child[i] = 1 / (1 + np.exp(-child[i]))
            return child,length_child+1
    elif tree[index] == 'INL1':
        return data[2],1
    elif tree[index] == 'PHC1':
        return data[3],1
    elif tree[index] == 'PLSC1':
        return data[4],1
    elif tree[index] == 'INC1':
        return data[5],1
    elif tree[index] == 'FOC1':
        return data[6],1
    elif tree[index] == 'PIP1':
        return data[7],1
    elif tree[index] == 'FC11':
        return data[8],1
    elif tree[index] == 'FC12':
        return data[9],1
    elif tree[index] == 'INL2':
        return data[10],1
    elif tree[index] == 'PHC2':
        return data[11],1
    elif tree[index] == 'PLSC2':
        return data[12],1
    elif tree[index] == 'INC2':
        return data[13],1
    elif tree[index] == 'FOC2':
        return data[14],1
    elif tree[index] == 'PIP2':
        return data[15],1
    elif tree[index] == 'FC21':
        return data[16],1
    elif tree[index] == 'FC22':
        return data[17],1
    elif tree[index] == 'PTC':
        return data[18],1
    elif tree[index] == 'FTC':
        return data[19],1


def GP_evolve_R(data, tree_R): # genetic programming evolved sequencing rule
    transshipment,length_tree = treeNode_R(tree_R, 0, data)  # todo: actually, this should be used for sequencing rule
    return transshipment


def treeNode_R(tree, index, data):
    if tree[index].arity == 2:
        left, length_left = treeNode_R(tree, index + 1, data)
        right, length_right = treeNode_R(tree, index + length_left + 1, data)
        if tree[index].name == 'add':
            return safe_add(left, right), length_left+length_right+1
        elif tree[index].name == 'subtract':
            return safe_subtract(left, right), length_left+length_right+1
        elif tree[index].name == 'multiply':
            return safe_multiply(left, right), length_left+length_right+1
        elif tree[index].name == 'protected_div':
            return protected_div(left, right), length_left+length_right+1
        elif tree[index].name == 'maximum':
            return np.maximum(left, right), length_left+length_right+1
        elif tree[index].name == 'minimum':
            return np.minimum(left, right), length_left+length_right+1
    elif tree[index].arity == 1:
        child, length_child = treeNode_R(tree, index + 1, data)
        if tree[index].name == 'lf':  # add by mengxu 2022.11.08
            if isinstance(child, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-child)),length_child+1
            else:
                for i in range(len(child)):
                    child[i] = 1 / (1 + np.exp(-child[i]))
                return child,length_child+1
        elif tree[index].name == 'protected_sqrt':
            return protected_sqrt(child),length_child+1
        elif tree[index].name == 'square':
            return safe_square(child),length_child+1
    elif tree[index].arity == 0:
        if tree[index].name == 'INL1':
            return data[2],1
        elif tree[index].name == 'PHC1':
            return data[3],1
        elif tree[index].name == 'PLSC1':
            return data[4],1
        elif tree[index].name == 'INC1':
            return data[5],1
        elif tree[index].name == 'FOC1':
            return data[6],1
        elif tree[index].name == 'PIP1':
            return data[7],1
        elif tree[index].name == 'FC11':
            return data[8],1
        elif tree[index].name == 'FC12':
            return data[9],1
        elif tree[index].name == 'INL2':
            return data[10],1
        elif tree[index].name == 'PHC2':
            return data[11],1
        elif tree[index].name == 'PLSC2':
            return data[12],1
        elif tree[index].name == 'INC2':
            return data[13],1
        elif tree[index].name == 'FOC2':
            return data[14],1
        elif tree[index].name == 'PIP2':
            return data[15],1
        elif tree[index].name == 'FC21':
            return data[16],1
        elif tree[index].name == 'FC22':
            return data[17],1
        elif tree[index].name == 'PTC':
            return data[18],1
        elif tree[index].name == 'FTC':
            return data[19],1


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x = np.where(np.isinf(x) | np.isnan(x), 1, x)
        else:
            x = 1 if np.isinf(x) or np.isnan(x) else x
    return x

def protected_sqrt(x):
    if x > 0:
        value = np.sqrt(x)
        value = np.inf if np.isnan(value) else value
    else:
        value = 0.0
    return value


def safe_multiply(val1, val2):
    try:
        # Ensure the values are floats
        val1 = float(val1)
        val2 = float(val2)

        # Check for infinity or NaN
        if np.isinf(val1) or np.isnan(val1) or np.isinf(val2) or np.isnan(val2):
            return np.inf  # or some safe value

        return val1 * val2
    except (ValueError, TypeError):
        # Handle cases where conversion to float fails
        return np.inf  # or some default safe value


def safe_subtract(a, b):
    with np.errstate(over='ignore', invalid='ignore'):  # Suppress overflow warnings
        value = np.subtract(a, b)
        if np.isinf(value) or np.isnan(value):
            value = np.inf  # Set overflowed or invalid results to infinity
    return value


def safe_add(a, b):
    with np.errstate(over='ignore', invalid='ignore'):  # Suppress overflow warnings
        value = np.add(a, b)
        if np.isinf(value) or np.isnan(value):
            value = np.inf  # Set overflowed or invalid results to infinity
    return value


def safe_square(a):
    with np.errstate(over='ignore', invalid='ignore'):  # Suppress overflow warnings
        try:
            a = float(a)
            if np.isinf(a) or np.isnan(a):
                return np.inf
            return np.square(a)
        except (ValueError, TypeError):
            # Handle cases where conversion to float fails
            return np.inf  # or some default safe value
    return value


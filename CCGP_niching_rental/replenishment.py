import math

import simpy
import random
import numpy as np
import math
from deap import gp

'''
this module contains the site replenishment rules used in the experiment
sequencing agents may choose to follow one of following rules
or choose to use trained parameters for decision-making
'''


def GP_pair_S_test(state, tree_S): #data is the state
    inventory_replenishment = treeNode_S_test(tree_S, 0, state)  # todo: actually, this should be used for sequencing rule
    return inventory_replenishment


def treeNode_S_test(tree, index, data):
    if tree[index] == 'add':
        return safe_add(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
        # return treeNode_S_test(tree, index+1, data) + treeNode_S_test(tree, index+2, data)
    elif tree[index] == 'subtract':
        # return treeNode_S_test(tree, index + 1, data) - treeNode_S_test(tree, index + 2, data)
        return safe_subtract(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'multiply':
        # return treeNode_S_test(tree, index + 1, data) * treeNode_S_test(tree, index + 2, data)
        return safe_multiply(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'protected_div':
        return protected_div(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'maximum':
        return np.maximum(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'minimum':
        return np.minimum(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'protected_sqrt':
        return protected_sqrt(treeNode_S_test(tree, index+1, data))
    elif tree[index] == 'square':
        return safe_square(treeNode_S_test(tree, index + 1, data))
        # return np.square(treeNode_S_test(tree, index+1, data))
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        ref = treeNode_S_test(tree, index+1, data)
        if isinstance(ref, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-ref))
        else:
            for i in range(len(ref)):
                ref[i] = 1 / (1 + np.exp(-ref[i]))
            return ref
    elif tree[index] == 'INL':
        return data[0]
    elif tree[index] == 'PHC':
        return data[1]
    elif tree[index] == 'PLSC':
        return data[2]
    elif tree[index] == 'INC':
        return data[3]
    elif tree[index] == 'FOC':
        return data[4]
    elif tree[index] == 'PIP':
        return data[5]
    elif tree[index] == 'FC1':
        return data[6]
    elif tree[index] == 'FC2':
        return data[7]
    elif tree[index] == 'PTC':
        return data[8]
    elif tree[index] == 'FTC':
        return data[9]

def GP_evolve_S(data, tree_S): # genetic programming evolved sequencing rule
    inventory_replenishment = treeNode_S(tree_S, 0, data)
    return inventory_replenishment

def treeNode_S(tree, index, data):
    if tree[index].arity == 2:
        if tree[index].name == 'add':
            return safe_add(treeNode_S(tree, index + 1, data), treeNode_S(tree, index + 2, data))
            # return treeNode_S(tree, index+1, data) + treeNode_S(tree, index+2, data)
        elif tree[index].name == 'subtract':
            # return treeNode_S(tree, index + 1, data) - treeNode_S(tree, index + 2, data)
            return safe_subtract(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
        elif tree[index].name == 'multiply':
            # return treeNode_S(tree, index + 1, data) * treeNode_S(tree, index + 2, data)
            return safe_multiply(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
        elif tree[index].name == 'protected_div':
            return protected_div(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
        elif tree[index].name == 'maximum':
            return np.maximum(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
        elif tree[index].name == 'minimum':
            return np.minimum(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            ref = treeNode_S(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref))
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref
        elif tree[index].name == 'protected_sqrt':
            return protected_sqrt(treeNode_S(tree, index + 1, data))
        elif tree[index].name == 'square':
            return safe_square(treeNode_S(tree, index + 1, data))
            # return np.square(treeNode_S(tree, index + 1, data))
    elif tree[index].arity == 0:
        if tree[index].name == 'INL':
            return data[0]
        elif tree[index].name == 'PHC':
            return data[1]
        elif tree[index].name == 'PLSC':
            return data[2]
        elif tree[index].name == 'INC':
            return data[3]
        elif tree[index].name == 'FOC':
            return data[4]
        elif tree[index].name == 'PIP':
            return data[5]
        elif tree[index].name == 'FC1':
            return data[6]
        elif tree[index].name == 'FC2':
            return data[7]
        elif tree[index].name == 'PTC':
            return data[8]
        elif tree[index].name == 'FTC':
            return data[9]

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


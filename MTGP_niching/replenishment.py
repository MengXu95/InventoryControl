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
    inventory_replenishment, length_tree = treeNode_S_test(tree_S, 0, state)  # todo: actually, this should be used for sequencing rule
    return inventory_replenishment


def treeNode_S_test(tree, index, data):
    if tree[index] == 'add':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return safe_add(left, right), length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return safe_subtract(left, right), length_left+length_right+1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return safe_multiply(left, right), length_left+length_right+1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return protected_div(left, right), length_left+length_right+1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return np.maximum(left, right), length_left+length_right+1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_S_test(tree, index + 1, data)
        right, length_right = treeNode_S_test(tree, index + length_left + 1, data)
        return np.minimum(left, right), length_left+length_right+1
    elif tree[index] == 'protected_sqrt':
        child, length_child = treeNode_S_test(tree, index + 1, data)
        return protected_sqrt(child),length_child+1
    elif tree[index] == 'square':
        child, length_child = treeNode_S_test(tree, index + 1, data)
        return safe_square(child),length_child+1
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        child, length_child = treeNode_S_test(tree, index + 1, data)
        if isinstance(child, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-child)),length_child+1
        else:
            for i in range(len(child)):
                child[i] = 1 / (1 + np.exp(-child[i]))
            return child,length_child+1
    elif tree[index] == 'INL':
        return data[0],1
    elif tree[index] == 'PHC':
        return data[1],1
    elif tree[index] == 'PLSC':
        return data[2],1
    elif tree[index] == 'INC':
        return data[3],1
    elif tree[index] == 'FOC':
        return data[4],1
    elif tree[index] == 'PIP':
        return data[5],1
    elif tree[index] == 'FC1':
        return data[6],1
    elif tree[index] == 'FC2':
        return data[7],1
    elif tree[index] == 'PTC':
        return data[8],1
    elif tree[index] == 'FTC':
        return data[9],1

def GP_evolve_S(data, tree_S): # genetic programming evolved sequencing rule
    inventory_replenishment, length_tree = treeNode_S(tree_S, 0, data)
    return inventory_replenishment

def treeNode_S(tree, index, data):
    if tree[index].arity == 2:
        left, length_left = treeNode_S(tree, index + 1, data)
        right, length_right = treeNode_S(tree, index + length_left + 1, data)
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
        child, length_child = treeNode_S(tree, index + 1, data)
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            if isinstance(child, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-child)),length_child+1
            else:
                for i in range(len(child)):
                    child[i] = 1 / (1 + np.exp(-child[i]))
                    # print(ref[i])
                return child,length_child+1
        elif tree[index].name == 'protected_sqrt':
            return protected_sqrt(child),length_child+1
        elif tree[index].name == 'square':
            return safe_square(child),length_child+1
    elif tree[index].arity == 0:
        if tree[index].name == 'INL':
            return data[0],1
        elif tree[index].name == 'PHC':
            return data[1],1
        elif tree[index].name == 'PLSC':
            return data[2],1
        elif tree[index].name == 'INC':
            return data[3],1
        elif tree[index].name == 'FOC':
            return data[4],1
        elif tree[index].name == 'PIP':
            return data[5],1
        elif tree[index].name == 'FC1':
            return data[6],1
        elif tree[index].name == 'FC2':
            return data[7],1
        elif tree[index].name == 'PTC':
            return data[8],1
        elif tree[index].name == 'FTC':
            return data[9],1

def is_valid(tree): # Returns Boolean indicating whether the tree is dimensionally valid
    _, dims = treeNode_S_with_units(tree, 0)
    is_valid = not np.array_equal(dims, np.array([np.inf,np.inf]))
    #print(f"Tree: {self.gpTree}, is_valid: {is_valid}")
    return is_valid
def treeNode_S_with_units(tree, index):
    if tree[index].arity == 2:
        length_left, dim_left = treeNode_S_with_units(tree, index + 1)
        length_right, dim_right = treeNode_S_with_units(tree, index + length_left + 1)
        if tree[index].name == 'add':
            if np.array_equal(dim_left, dim_right):  # Works even if both are inf
                return length_left + length_right + 1, dim_left
            else:  # Dimension mismatch
                return length_left + length_right + 1, np.array([np.inf, np.inf])
        elif tree[index].name == 'subtract':
            if np.array_equal(dim_left, dim_right):  # Works even if both are inf
                return length_left + length_right + 1, dim_left
            else:  # Dimension mismatch
                return length_left + length_right + 1, np.array([np.inf, np.inf])
        elif tree[index].name == 'multiply':
            return length_left + length_right + 1, dim_left + dim_right if not np.array_equal(dim_right, np.array(
                [np.inf, np.inf])) else np.array([np.inf, np.inf])
        elif tree[index].name == 'protected_div':
            return length_left + length_right + 1, dim_left - dim_right if not np.array_equal(dim_right, np.array(
                [np.inf, np.inf])) else np.array([np.inf, np.inf])
        elif tree[index].name == 'maximum':
            return length_left + length_right + 1, dim_left if np.array_equal(dim_left, dim_right) else np.array(
                [np.inf, np.inf])
        elif tree[index].name == 'minimum':
            return length_left + length_right + 1, dim_left if np.array_equal(dim_left, dim_right) else np.array(
                [np.inf, np.inf])
    elif tree[index].arity == 1:
        length_child, dim_child = treeNode_S_with_units(tree, index + 1)
        return length_child + 1, dim_child
    elif tree[index].arity == 0:
        # dims: [quantity, cost]
        if tree[index].name == 'INL':
            return 1, np.array([1, 0])
        elif tree[index].name == 'PHC':
            return 1, np.array([0, 1])
        elif tree[index].name == 'PLSC':
            return 1, np.array([0, 1])
        elif tree[index].name == 'INC':
            return 1, np.array([1, 0])
        elif tree[index].name == 'FOC':
            return 1, np.array([0, 1])
        elif tree[index].name == 'PIP':
            return 1, np.array([1, 0])
        elif tree[index].name == 'FC1':
            return 1, np.array([1, 0])
        elif tree[index].name == 'FC2':
            return 1, np.array([1, 0])
        elif tree[index].name == 'PTC':
            return 1, np.array([0, 1])
        elif tree[index].name == 'FTC':
            return 1, np.array([0, 1])

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

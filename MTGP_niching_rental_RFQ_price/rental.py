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


def GP_pair_rental_test(state, tree_S): #data is the state
    rental_priority,length_tree = treeNode_rental_test(tree_S, 0, state)  # todo: actually, this should be used for sequencing rule
    return rental_priority


def treeNode_rental_test(tree, index, data):
    if tree[index] == 'add':
        left, length_left = treeNode_rental_test(tree, index + 1, data)
        right, length_right = treeNode_rental_test(tree, index + length_left + 1, data)
        return safe_add(left, right), length_left+length_right+1
    elif tree[index] == 'subtract':
        left, length_left = treeNode_rental_test(tree, index + 1, data)
        right, length_right = treeNode_rental_test(tree, index + length_left + 1, data)
        return safe_subtract(left, right), length_left+length_right+1
    elif tree[index] == 'multiply':
        left, length_left = treeNode_rental_test(tree, index + 1, data)
        right, length_right = treeNode_rental_test(tree, index + length_left + 1, data)
        return safe_multiply(left, right), length_left+length_right+1
    elif tree[index] == 'protected_div':
        left, length_left = treeNode_rental_test(tree, index + 1, data)
        right, length_right = treeNode_rental_test(tree, index + length_left + 1, data)
        return protected_div(left, right), length_left+length_right+1
    elif tree[index] == 'maximum':
        left, length_left = treeNode_rental_test(tree, index + 1, data)
        right, length_right = treeNode_rental_test(tree, index + length_left + 1, data)
        return np.maximum(left, right), length_left+length_right+1
    elif tree[index] == 'minimum':
        left, length_left = treeNode_rental_test(tree, index + 1, data)
        right, length_right = treeNode_rental_test(tree, index + length_left + 1, data)
        return np.minimum(left, right), length_left+length_right+1
    elif tree[index] == 'protected_sqrt':
        child, length_child = treeNode_rental_test(tree, index + 1, data)
        return protected_sqrt(child),length_child+1
    elif tree[index] == 'square':
        child, length_child = treeNode_rental_test(tree, index + 1, data)
        return safe_square(child),length_child+1
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        child, length_child = treeNode_rental_test(tree, index + 1, data)
        if isinstance(child, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-child)),length_child+1
        else:
            for i in range(len(child)):
                child[i] = 1 / (1 + np.exp(-child[i]))
            return child,length_child+1
    elif tree[index] == 'CRENT':
        return data[0],1
    elif tree[index] == 'RP':
        return data[1],1
    elif tree[index] == 'RCAP':
        return data[2],1
    elif tree[index] == 'RDUR':
        return data[3],1
    elif tree[index] == 'TREQ':
        return data[4],1

def GP_evolve_rental(data, tree_S): # genetic programming evolved sequencing rule
    rental_priority,length_tree = treeNode_rental(tree_S, 0, data)
    return rental_priority

def is_valid(tree): # Returns Boolean indicating whether the tree is dimensionally valid
    _, dims = treeNode_rental_with_units(tree, 0)
    is_valid = not np.array_equal(dims, np.array([np.inf,np.inf]))
    #print(f"Tree: {self.gpTree}, is_valid: {is_valid}")
    return is_valid

def treeNode_rental_with_units(tree, index):
    if tree[index].arity == 2:
        length_left, dim_left = treeNode_rental_with_units(tree, index + 1)
        length_right, dim_right = treeNode_rental_with_units(tree, index + length_left + 1)
        if tree[index].name == 'add':
            if np.array_equal(dim_left, dim_right):  # Works even if both are inf
                return length_left + length_right + 1, dim_left
            else:  # Dimension mismatch
                return length_left + length_right + 1, np.array([np.inf, np.inf, np.inf])
        elif tree[index].name == 'subtract':
            if np.array_equal(dim_left, dim_right):  # Works even if both are inf
                return length_left + length_right + 1, dim_left
            else:  # Dimension mismatch
                return length_left + length_right + 1, np.array([np.inf, np.inf, np.inf])
        elif tree[index].name == 'multiply':
            return length_left + length_right + 1, dim_left + dim_right if not np.array_equal(dim_right, np.array(
                [np.inf, np.inf, np.inf])) else np.array([np.inf, np.inf, np.inf])
        elif tree[index].name == 'protected_div':
            return length_left + length_right + 1, dim_left - dim_right if not np.array_equal(dim_right, np.array(
                [np.inf, np.inf, np.inf])) else np.array([np.inf, np.inf, np.inf])
        elif tree[index].name == 'maximum':
            return length_left + length_right + 1, dim_left if np.array_equal(dim_left, dim_right) else np.array(
                [np.inf, np.inf, np.inf])
        elif tree[index].name == 'minimum':
            return length_left + length_right + 1, dim_left if np.array_equal(dim_left, dim_right) else np.array(
                [np.inf, np.inf, np.inf])
    elif tree[index].arity == 1:
        length_child, dim_child = treeNode_rental_with_units(tree, index + 1)
        return length_child + 1, dim_child
    elif tree[index].arity == 0:
        if tree[index].name == 'CRENT':
            return 1,np.array([1,0,0])
        elif tree[index].name == 'RP':
            return 1,np.array([0,1,0])
        elif tree[index].name == 'RCAP':
            return 1,np.array([1,0,0])
        elif tree[index].name == 'RDUR':
            return 1,np.array([0,0,1])
        elif tree[index].name == 'TREQ':
            return 1,np.array([1,0,0])
def treeNode_rental(tree, index, data):
    if tree[index].arity == 2:
        left, length_left = treeNode_rental(tree, index + 1, data)
        right, length_right = treeNode_rental(tree, index + length_left + 1, data)
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
        child, length_child = treeNode_rental(tree, index + 1, data)
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
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
        if tree[index].name == 'CRENT':
            return data[0],1
        elif tree[index].name == 'RP':
            return data[1],1
        elif tree[index].name == 'RCAP':
            return data[2],1
        elif tree[index].name == 'RDUR':
            return data[3],1
        elif tree[index].name == 'TREQ':
            return data[4],1

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


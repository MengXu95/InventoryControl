import copy
import random
import numpy as np

from deap import gp, creator
from deap import tools



def init_primitives(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)

    # terminals for sequencing and routing in my paper
    # pset.addTerminal(str('INL1'))  # add by mengxu for 2 sites
    # pset.addTerminal(str('INL2'))  # add by mengxu
    # pset.addTerminal(str('FC11'))  # add by mengxu
    # pset.addTerminal(str('FC12'))  # add by mengxu
    # pset.addTerminal(str('FC21'))  # add by mengxu
    # pset.addTerminal(str('FC22'))  # add by mengxu
    # pset.addTerminal(str('PIP1'))  # add by mengxu for example L = 2
    # pset.addTerminal(str('PIP2'))  # add by mengxu for example L = 2

    pset.addTerminal(str('INL1'))  # add by mengxu for 1 site
    pset.addTerminal(str('FC11'))  # add by mengxu
    pset.addTerminal(str('FC12'))  # add by mengxu
    pset.addTerminal(str('PIP1'))  # add by mengxu for example L = 2


def init_primitives_replenishment(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(np.square, 1)

    pset.addTerminal(str('INL'))  # inventory level
    pset.addTerminal(str('PHC'))  # per unit holding cost
    pset.addTerminal(str('PLSC'))  # per unit lost sales cost
    pset.addTerminal(str('INC'))  # inventory capacities
    pset.addTerminal(str('FOC'))  # Fixed order costs per order
    pset.addTerminal(str('PIP'))  # pipeline: the previous quantity we ordered and arrived at this time
    pset.addTerminal(str('FC1'))  # forcast 1 for example L = 2
    pset.addTerminal(str('FC2'))  # forcast 2 for example L = 2
    pset.addTerminal(str('PTC'))  # per unit cost transshipment
    pset.addTerminal(str('FTC'))  # fixed cost per transshipment

def init_primitives_transshipment(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(np.square, 1)

    #site 1 related
    pset.addTerminal(str('INL1'))  # inventory level
    pset.addTerminal(str('PHC1'))  # per unit holding cost
    pset.addTerminal(str('PLSC1'))  # per unit lost sales cost
    pset.addTerminal(str('INC1'))  # inventory capacities
    pset.addTerminal(str('FOC1'))  # Fixed order costs per order
    pset.addTerminal(str('PIP1'))  # pipeline: the previous quantity we ordered and arrived at this time
    pset.addTerminal(str('FC11'))  # forcast 1 for example L = 2
    pset.addTerminal(str('FC12'))  # forcast 2 for example L = 2
    # site 2 related
    pset.addTerminal(str('INL2'))  # inventory level
    pset.addTerminal(str('PHC2'))  # per unit holding cost
    pset.addTerminal(str('PLSC2'))  # per unit lost sales cost
    pset.addTerminal(str('INC2'))  # inventory capacities
    pset.addTerminal(str('FOC2'))  # Fixed order costs per order
    pset.addTerminal(str('PIP2'))  # pipeline: the previous quantity we ordered and arrived at this time
    pset.addTerminal(str('FC21'))  # forcast 1 for example L = 2
    pset.addTerminal(str('FC22'))  # forcast 2 for example L = 2
    #transshipment related
    pset.addTerminal(str('PTC'))  # per unit cost transshipment
    pset.addTerminal(str('FTC'))  # fixed cost per transshipment


def init_primitives_rental(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(np.square, 1)

    pset.addTerminal(str('CRENT'))  # total current rental
    pset.addTerminal(str('RP'))  # rental price per rental choice
    pset.addTerminal(str('RCAP'))  # rental capacity per rental choice
    pset.addTerminal(str('RDUR'))  # rental duration per rental choice
    pset.addTerminal(str('TREQ'))  # total inventory requirement from all sites


def init_primitives_RFQ_predict(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(np.square, 1)

    pset.addTerminal(str('RFQ'))  # EFQ demand
    pset.addTerminal(str('TUD'))  # inventory level of our company


def lf(x): # add by mengxu 2022.11.08
    return 1 / (1 + np.exp(-x))


def init_toolbox(toolbox, pset, num_tree):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6) # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=num_tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate",lim_xmate)
    toolbox.register("mutate",lim_xmut_onetree,expr=toolbox.expr_mut)


def init_toolbox_two_pset(toolbox, pset1, pset2, num_tree=2):
    # Define the individual type with a list of trees and the appropriate fitness function
    creator.create("Individual", list, fitness=creator.FitnessMin, pset1=pset1, pset2=pset2)

    # Register the expression generation functions for each pset
    toolbox.register("expr1", gp.genHalfAndHalf, pset=pset1, min_=1, max_=6)
    toolbox.register("expr2", gp.genHalfAndHalf, pset=pset2, min_=1, max_=6)

    # Register tree initialization for each pset
    toolbox.register("tree1", tools.initIterate, gp.PrimitiveTree, toolbox.expr1)
    toolbox.register("tree2", tools.initIterate, gp.PrimitiveTree, toolbox.expr2)

    # Register individual creation, using both trees
    def generate_individual():
        return creator.Individual([toolbox.tree1(), toolbox.tree2()])

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the compile function for each pset
    toolbox.register("compile1", gp.compile, pset=pset1)
    toolbox.register("compile2", gp.compile, pset=pset2)

    # Register mutation with the expression function from the first pset
    toolbox.register("expr_mut1", gp.genHalfAndHalf, pset=pset1, min_=2, max_=4)
    toolbox.register("expr_mut2", gp.genHalfAndHalf, pset=pset2, min_=2, max_=4)

    # Register mating and mutation functions
    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut, expr1=toolbox.expr_mut1, expr2=toolbox.expr_mut2)


def init_toolbox_three_pset(toolbox, pset1, pset2, pset3, num_tree=3):
    # Define the individual type with a list of trees and the appropriate fitness function
    creator.create("Individual", list, fitness=creator.FitnessMin, pset1=pset1, pset2=pset2, pset3=pset3)

    # Register the expression generation functions for each pset
    toolbox.register("expr1", gp.genHalfAndHalf, pset=pset1, min_=1, max_=6)
    toolbox.register("expr2", gp.genHalfAndHalf, pset=pset2, min_=1, max_=6)
    toolbox.register("expr3", gp.genHalfAndHalf, pset=pset3, min_=1, max_=6)

    # Register tree initialization for each pset
    toolbox.register("tree1", tools.initIterate, gp.PrimitiveTree, toolbox.expr1)
    toolbox.register("tree2", tools.initIterate, gp.PrimitiveTree, toolbox.expr2)
    toolbox.register("tree3", tools.initIterate, gp.PrimitiveTree, toolbox.expr3)

    # Register individual creation, using both trees
    def generate_individual():
        return creator.Individual([toolbox.tree1(), toolbox.tree2(), toolbox.tree3()])

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the compile function for each pset
    toolbox.register("compile1", gp.compile, pset=pset1)
    toolbox.register("compile2", gp.compile, pset=pset2)
    toolbox.register("compile3", gp.compile, pset=pset3)

    # Register mutation with the expression function from the first pset
    toolbox.register("expr_mut1", gp.genHalfAndHalf, pset=pset1, min_=2, max_=4)
    toolbox.register("expr_mut2", gp.genHalfAndHalf, pset=pset2, min_=2, max_=4)
    toolbox.register("expr_mut3", gp.genHalfAndHalf, pset=pset3, min_=2, max_=4)

    # Register mating and mutation functions
    toolbox.register("mate", lim_xmate_three_tree)
    toolbox.register("mutate", lim_xmut_three_tree, expr1=toolbox.expr_mut1, expr2=toolbox.expr_mut2, expr3=toolbox.expr_mut3)


def maxheight(v):
    return max(i.height for i in v)


# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    MAX_HEIGHT = 8 #todo: only for test, need to be the same with original GPFC.py
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > MAX_HEIGHT: # original
            new_inds[i] = random.choice(keep_inds)
        # while maxheight(new_inds) > MAX_HEIGHT: # modified on 2024.8.22
        #     new_inds[i] = list(func(*args, **kwargs))[i]
    return new_inds


def xmate_three_tree(ind1, ind2):
    i1 = random.randrange(len(ind1))
    #todo: I think this is not same with my MTGP, as only the same type of tree can be used to do crossover
    ind1[i1], ind2[i1] = gp.cxOnePoint(ind1[i1], ind2[i1])

    # #exchange the other two tree
    # if len(ind1) == 2:
    #     i2 = 1 - i1 # only for individual with two tree
    #     ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    return ind1, ind2

def lim_xmate_three_tree(ind1, ind2):
    return wrap(xmate_three_tree, ind1, ind2)


def xmut_three_tree(ind, expr1, expr2, expr3):
    # Randomly select which tree to mutate (0 for the first tree, 1 for the second tree)
    i1 = random.randrange(len(ind))

    # Select the correct expr and pset based on the tree index
    if i1 == 0:
        indx = gp.mutUniform(ind[i1], expr1, pset=ind.pset1)
    elif i1 == 1:
        indx = gp.mutUniform(ind[i1], expr2, pset=ind.pset2)
    else:
        indx = gp.mutUniform(ind[i1], expr3, pset=ind.pset3)

    # Replace the mutated tree in the individual
    ind[i1] = indx[0]

    return ind,


def lim_xmut_three_tree(ind, expr1, expr2, expr3):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut_three_tree, ind, expr1=expr1, expr2=expr2, expr3=expr3)
    # print(res)
    return res

# the following is modified by mengxu
def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    # i2 = random.randrange(len(ind2))
    #todo: I think this is not same with my MTGP, as only the same type of tree can be used to do crossover
    ind1[i1], ind2[i1] = gp.cxOnePoint(ind1[i1], ind2[i1])

    #exchange the other tree
    if len(ind1) == 2:
        i2 = 1 - i1 # only for individual with two tree
        ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    return ind1, ind2

def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def xmut(ind, expr1, expr2):
    # Randomly select which tree to mutate (0 for the first tree, 1 for the second tree)
    i1 = random.randrange(len(ind))

    # Select the correct expr and pset based on the tree index
    if i1 == 0:
        indx = gp.mutUniform(ind[i1], expr1, pset=ind.pset1)
    else:
        indx = gp.mutUniform(ind[i1], expr2, pset=ind.pset2)

    # Replace the mutated tree in the individual
    ind[i1] = indx[0]

    return ind,


def lim_xmut(ind, expr1, expr2):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr1=expr1, expr2=expr2)
    # print(res)
    return res

def xmut_onetree(ind, expr):
    # Randomly select which tree to mutate (0 for the first tree, 1 for the second tree)
    i1 = random.randrange(len(ind))

    # Select the correct expr and pset based on the tree index
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)

    # Replace the mutated tree in the individual
    ind[i1] = indx[0]

    return ind,


def lim_xmut_onetree(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut_onetree, ind, expr=expr)
    # print(res)
    return res


def add_abs(a, b):
    return np.abs(np.add(a, b))


def sub_abs(a, b):
    return np.abs(np.subtract(a, b))


def mt_if(a, b, c):
    return np.where(a < 0, b, c)


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def protected_sqrt(x):
    if x > 0:
        value = np.sqrt(x)
    else:
        value = 0.0
    return value
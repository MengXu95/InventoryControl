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

def lf(x): # add by mengxu 2022.11.08
    return 1 / (1 + np.exp(-x))


def init_toolbox(toolbox, pset, num_tree):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6) # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=num_tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=4)

    toolbox.register("mate",lim_xmate)
    toolbox.register("mutate",lim_xmut,expr=toolbox.expr_mut)


def init_toolbox_two_pset(toolbox, pset1, pset2):
    # Define individual types for each subpopulation
    creator.create("IndividualSubpop1", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset1)
    creator.create("IndividualSubpop2", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset2)

    # Register the expression generation functions for each pset
    toolbox.register("expr1", gp.genHalfAndHalf, pset=pset1, min_=1, max_=6)
    toolbox.register("expr2", gp.genHalfAndHalf, pset=pset2, min_=1, max_=6)

    # Register tree initialization for each subpopulation
    toolbox.register("tree1", tools.initIterate, gp.PrimitiveTree, toolbox.expr1)
    toolbox.register("tree2", tools.initIterate, gp.PrimitiveTree, toolbox.expr2)

    # Register individual creation functions for each subpopulation
    toolbox.register("individual1", tools.initIterate, creator.IndividualSubpop1, toolbox.tree1)
    toolbox.register("individual2", tools.initIterate, creator.IndividualSubpop2, toolbox.tree2)

    # Register subpopulation creation functions
    toolbox.register("subpop1", tools.initRepeat, list, toolbox.individual1)
    toolbox.register("subpop2", tools.initRepeat, list, toolbox.individual2)

    # Combine both subpopulations into a single population list
    def generate_population(n):
        subpop1 = toolbox.subpop1(n // 2)
        subpop2 = toolbox.subpop2(n // 2)
        return [subpop1, subpop2]

    toolbox.register("population", generate_population)

    # Register the compile functions for each pset
    toolbox.register("compile1", gp.compile, pset=pset1)
    toolbox.register("compile2", gp.compile, pset=pset2)

    # Register mutation functions for each subpopulation
    toolbox.register("expr_mut1", gp.genHalfAndHalf, pset=pset1, min_=2, max_=4)
    toolbox.register("expr_mut2", gp.genHalfAndHalf, pset=pset2, min_=2, max_=4)

    # Register mating and mutation functions, specific to the subpopulation
    toolbox.register("mate1", gp.cxOnePoint)
    toolbox.register("mate2", gp.cxOnePoint)
    toolbox.register("mutate1", lim_xmut, expr=toolbox.expr_mut1)
    toolbox.register("mutate2", lim_xmut, expr=toolbox.expr_mut2)

    # Register overall mating and mutation functions
    def mate(ind1, ind2):
        if isinstance(ind1, creator.IndividualSubpop1) and isinstance(ind2, creator.IndividualSubpop1):
            return toolbox.mate1(ind1, ind2)
        elif isinstance(ind1, creator.IndividualSubpop2) and isinstance(ind2, creator.IndividualSubpop2):
            return toolbox.mate2(ind1, ind2)
        else:
            raise ValueError("Mating should occur between individuals of the same subpopulation.")

    def mutate(ind):
        if isinstance(ind, creator.IndividualSubpop1):
            return toolbox.mutate1(ind)
        elif isinstance(ind, creator.IndividualSubpop2):
            return toolbox.mutate2(ind)
        else:
            raise ValueError("Mutation should be applied to an individual of a recognized subpopulation.")

    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)



def maxheight(v):
    return max(i.height for i in v)


# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    MAX_HEIGHT = 8 #todo: only for test, need to be the same with original GPFC.py
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    if maxheight(new_inds) > MAX_HEIGHT:  # original
        new_inds = (random.choice(keep_inds),)
    # while maxheight(new_inds) > MAX_HEIGHT: # modified by mengxu 2024.8.24
    #     new_inds = list(func(*args, **kwargs))
    return new_inds

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

# def xmate(ind1, ind2):
#     i1 = random.randrange(len(ind1))
#     i2 = random.randrange(len(ind2))
#     ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
#     return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)

def xmut(ind, expr):
    indx = gp.mutUniform(ind, expr, pset=ind.pset)
    ind = indx[0]
    return ind,


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
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

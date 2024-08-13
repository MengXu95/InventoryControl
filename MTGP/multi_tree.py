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

    pset.addTerminal(str('INL1'))  # add by mengxu for 2 sites
    pset.addTerminal(str('FC11'))  # add by mengxu
    pset.addTerminal(str('FC12'))  # add by mengxu
    pset.addTerminal(str('PIP1'))  # add by mengxu for example L = 2






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
    #toolbox.register("mate", xmate)
    #toolbox.register("mutate", xmut, expr=toolbox.expr_mut)

    toolbox.register("mate",lim_xmate)
    toolbox.register("mutate",lim_xmut,expr=toolbox.expr_mut)



def maxheight(v):
    return max(i.height for i in v)


# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    MAX_HEIGHT = 8 #todo: only for test, need to be the same with original GPFC.py
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > MAX_HEIGHT:
            new_inds[i] = random.choice(keep_inds)
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
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr,pset=ind.pset)
    ind[i1] = indx[0]
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

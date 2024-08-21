import sys
import MTGP.LoadIndividual as mtload
import pygraphviz as pgv
from deap import gp
import pickle
import matplotlib.pyplot as plt
import networkx as nx

def graph(expr):
    """Construct the graph of a tree expression. The tree expression must be
    valid. It returns in order a node list, an edge list, and a dictionary of
    the per node labels. The node are represented by numbers, the edges are
    tuples connecting two nodes (number), and the labels are values of a
    dictionary for which keys are the node numbers.

    :param expr: A tree expression to convert into a graph.
    :returns: A node list, an edge list, and a dictionary of labels.

    The returned objects can be used directly to populate a
    `pygraphviz <http://networkx.lanl.gov/pygraphviz/>`_ graph::

        import pygraphviz as pgv

        # [...] Execution of code that produce a tree expression

        nodes, edges, labels = graph(expr)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw("tree.pdf")

    or a `NetworX <http://networkx.github.com/>`_ graph::

        import matplotlib.pyplot as plt
        import networkx as nx

        # [...] Execution of code that produce a tree expression

        nodes, edges, labels = graph(expr)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()


    .. note::

       We encourage you to use `pygraphviz
       <http://networkx.lanl.gov/pygraphviz/>`_ as the nodes might be plotted
       out of order when using `NetworX <http://networkx.github.com/>`_.
    """
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = transferToFunction(node) if isFunction(node) else node
        if isFunction(node):
            stack.append([i, 2])
        else:
            stack.append([i, 0])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels

def transferToFunction(node):
    if node == 'add':
        return '+'
    elif node == 'subtract':
        return '-'
    elif node == 'multiply':
        return '*'
    elif node == 'protected_div':
        return '/'
    elif node == 'maximum':
        return 'max'
    elif node == 'minimum':
        return 'min'
    else:
        return False

def isFunction(node):
    if node == 'add':
        return True
    elif node == 'subtract':
        return True
    elif node == 'multiply':
        return True
    elif node == 'protected_div':
        return True
    elif node == 'maximum':
        return True
    elif node == 'minimum':
        return True
    else:
        return False

if __name__ == "__main__":
#     dataSetName = str(sys.argv[1])
#     seedOfRun = int(sys.argv[2])
    dataSetName = 'sN2h_1_5b2'
    seedOfRun = 0

    # MTGP rule test, test the best rule obtained from all the generations
    # dict_best_MTGP_individuals = mtload.load_individual_from_gen(seedOfRun, dataSetName)
    with open('../MTGP_niching/train/scenario_' + str(dataSetName) + '/' + str(
        seedOfRun) + '_meng_individual_' + dataSetName + '.pkl',
          "rb") as fileName_individual:
        dict_best_MTGP_individuals = pickle.load(fileName_individual)

    individual = dict_best_MTGP_individuals.get(38)
    replenishment_rule_tree = individual[0]
    transshipment_rule_tree = individual[1]

    # sequencing tree
    nodes, edges, labels = graph(replenishment_rule_tree)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(dataSetName + "_" + str(seedOfRun) + "_replenishment_tree.pdf")

    # routing tree
    nodes, edges, labels = graph(transshipment_rule_tree)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(dataSetName + "_" + str(seedOfRun) + "_transshipment_tree.pdf")

import sys
import MTGP_niching.LoadIndividual as mtload
import pygraphviz as pgv
from deap import gp
import pickle
import matplotlib.pyplot as plt
import networkx as nx


def graph(expr):
    """
    Construct the graph of a tree expression. The tree expression must be
    valid. It returns in order a node list, an edge list, and a dictionary of
    the per node labels. The nodes are represented by numbers, the edges are
    tuples connecting two nodes (number), and the labels are values of a
    dictionary for which keys are the node numbers.

    :param expr: A tree expression to convert into a graph.
    :returns: A node list, an edge list, and a dictionary of labels.
    """
    nodes = list(range(len(expr)))
    edges = []
    labels = {}

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1

        # Transfer to appropriate function or label
        labels[i] = transferToFunction(node) if (isFunction1(node) or isFunction2(node)) else node

        # Handling both 1-input and 2-input functions
        if isFunction2(node):
            stack.append([i, 2])
        elif isFunction1(node):
            stack.append([i, 1])
        else:
            stack.append([i, 0])

        # Pop the stack when no more children are needed
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
    elif node == 'protected_sqrt':
        return 'sqrt'
    elif node == 'square':
        return '^2'
    else:
        return node


def isFunction2(node):
    # Functions that take 2 inputs
    return node in ['add', 'subtract', 'multiply', 'protected_div', 'maximum', 'minimum',
                    '+', '-', '*', '/', 'max', 'min']


def isFunction1(node):
    # Functions that take 1 input
    return node in ['protected_sqrt', 'square', 'sqrt', '^2']


if __name__ == "__main__":
    dataSetName = 'lN2h_1_5b2'
    seedOfRun = 4

    # Load the best individual from all generations
    with open('../MTGP_niching/train/scenario_' + str(dataSetName) + '/' + str(
            seedOfRun) + '_meng_individual_' + dataSetName + '.pkl', "rb") as fileName_individual:
        dict_best_MTGP_individuals = pickle.load(fileName_individual)

    individual = dict_best_MTGP_individuals.get(len(dict_best_MTGP_individuals) - 1)
    replenishment_rule_tree = individual[0]
    transshipment_rule_tree = individual[1]


    # Define a function to beautify the graph with different colors for leaf and non-leaf nodes
    def beautify_graph(nodes, edges, labels, file_name):
        g = pgv.AGraph(directed=True)  # Set directed=True to handle tree layout properly
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        # Apply beautification
        for i in nodes:
            n = g.get_node(i)

            # Check if the node is a leaf or non-leaf (function node)
            if isFunction1(labels[i]) or isFunction2(labels[i]):
                # Non-leaf node (function)
                n.attr["shape"] = "box"
                n.attr["style"] = "rounded, filled"
                n.attr["fillcolor"] = "#C6E9F4"  # Light blue for non-leaf nodes
            else:
                # Leaf node (terminal)
                n.attr["shape"] = "box"
                n.attr["style"] = "rounded, filled"
                n.attr["fillcolor"] = "#FFEBAD"  # Gold color for leaf nodes

            n.attr["fontname"] = "Arial"
            n.attr["label"] = labels[i]  # Add labels to nodes

        # Edge beautification
        for edge in edges:
            e = g.get_edge(edge[0], edge[1])
            e.attr["color"] = "black"  # Grey color for edges

        g.layout(prog="dot")  # Use the dot layout for better tree structure
        g.draw(file_name)


    # Plot replenishment tree
    nodes, edges, labels = graph(replenishment_rule_tree)
    beautify_graph(nodes, edges, labels, dataSetName + "_" + str(seedOfRun) + "_replenishment_tree.png")

    # Plot transshipment tree
    nodes, edges, labels = graph(transshipment_rule_tree)
    beautify_graph(nodes, edges, labels, dataSetName + "_" + str(seedOfRun) + "_transshipment_tree.png")

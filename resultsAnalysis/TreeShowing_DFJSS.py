import MTGP_niching.LoadIndividual as mtload
import pygraphviz as pgv
from deap import gp
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import re


def transferToFunction(node):
    """Maps a function name to a more common symbol."""
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
    """Checks if a node represents a function with 2 inputs."""
    return node in ['add', 'subtract', 'multiply', 'protected_div', 'maximum', 'minimum',
                    '+', '-', '*', '/', 'max', 'min']


def isFunction1(node):
    """Checks if a node represents a function with 1 input."""
    return node in ['protected_sqrt', 'square', 'sqrt', '^2']


def parse_expression(expr_str):
    """
    Parses a string expression and converts it into a list representing a tree.
    This function is a simplified parser for the given expression format.
    """
    # Use regular expressions to find all function names and arguments (including nested ones)
    matches = re.findall(r"(\w+)\(|'([^']*)'|\)", expr_str)

    expr_list = []
    stack = []

    for match in matches:
        func, term = match
        if func:
            # It's a function
            expr_list.append(func)
            if isFunction2(func):
                stack.append(2)
            elif isFunction1(func):
                stack.append(1)
        elif term:
            # It's a terminal
            expr_list.append(term)
            if stack:
                stack[-1] -= 1

        while stack and stack[-1] == 0:
            stack.pop()

    return expr_list


def graph(expr):
    """
    Constructs the graph of a tree expression from a list.
    Returns a node list, an edge list, and a dictionary of labels.
    """
    nodes = list(range(len(expr)))
    edges = []
    labels = {}

    stack = []
    for i, node in enumerate(expr):
        if stack:
            parent_index = stack[-1][0]
            edges.append((parent_index, i))
            stack[-1][1] -= 1

        labels[i] = transferToFunction(node)

        # Handle both 1-input and 2-input functions
        if isFunction2(node):
            stack.append([i, 2])
        elif isFunction1(node):
            stack.append([i, 1])
        else:
            # Terminals (leaves)
            stack.append([i, 0])

        # Pop the stack when no more children are needed
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


def beautify_graph(nodes, edges, labels, file_name):
    """
    Creates and saves a beautiful graph visualization with colored nodes.
    """
    g = pgv.AGraph(directed=True)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    for i in nodes:
        n = g.get_node(i)
        node_label = labels[i]

        if isFunction1(node_label) or isFunction2(node_label):
            n.attr["shape"] = "box"
            n.attr["style"] = "rounded, filled"
            n.attr["fillcolor"] = "#C6E9F4"  # Light blue for function nodes
        else:
            n.attr["shape"] = "box"
            n.attr["style"] = "rounded, filled"
            n.attr["fillcolor"] = "#FFEBAD"  # Gold for terminal nodes

        n.attr["fontname"] = "Arial"
        n.attr["label"] = node_label

    for edge in edges:
        e = g.get_edge(edge[0], edge[1])
        e.attr["color"] = "black"

    g.layout(prog="dot")
    g.draw(file_name)


if __name__ == "__main__":
    # Example usage with the provided string
    tree_string = "maximum(maximum(maximum(multiply(minimum('MWT', 'OWT'), minimum(minimum('MWT', 'NPT'), 'NPT')), 'WIQ'), multiply(subtract(protected_div('OWT', 'NIQ'), maximum('TIS', 'NIQ')), maximum(minimum('WIQ', 'NPT'), protected_div('MWT', 'WIQ')))), minimum(subtract(protected_div(minimum('SLACK', 'OWT'), maximum('MWT', 'TIS')), subtract(minimum(minimum('OWT', 'OWT'), 'WIQ'), minimum(add('NIQ', 'MWT'), protected_div('PT', 'MWT')))), protected_div(protected_div(multiply('WKR', 'OWT'), minimum('WKR', 'OWT')), maximum(protected_div('WIQ', multiply('OWT', 'NOR')), maximum('SLACK', 'SLACK')))))"

    # Parse the string into a list representation
    tree_list = parse_expression(tree_string)

    # Generate the graph and save it
    nodes, edges, labels = graph(tree_list)
    beautify_graph(nodes, edges, labels, "output_tree.png")

    print("Tree visualization saved to output_tree.png")
import os
import pickle
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter

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


def main(algo, dataSetName, num_runs=30):
    candidate_terminals_replenishment = ['INL', 'PHC', 'PLSC', 'INC', 'FOC', 'PIP', 'FC1', 'FC2', 'PTC', 'FTC',
                                         '+', '-', '*', '/', 'max', 'min', 'sqrt', '^2']
    candidate_terminals_transshipment = ['INL1', 'PHC1', 'PLSC1', 'INC1', 'FOC1', 'PIP1', 'FC11', 'FC12',
                                         'INL2', 'PHC2', 'PLSC2', 'INC2', 'FOC2', 'PIP2', 'FC21', 'FC22',
                                         'PTC', 'FTC', '+', '-', '*', '/', 'max', 'min', 'sqrt', '^2']

    def count_node_appearances(candidate_terminals, expr):
        nodes, edges, labels = graph(expr)
        node_counts = {terminal: 0 for terminal in candidate_terminals}  # Initialize all counts to 0
        actual_node_counts = Counter(labels.values())  # Count only the nodes present in the expression
        node_counts.update(actual_node_counts)  # Update with actual counts
        return node_counts

    replenishment_total_counts = {terminal: 0 for terminal in candidate_terminals_replenishment}
    transshipment_total_counts = {terminal: 0 for terminal in candidate_terminals_transshipment}

    for seedOfRun in range(1, num_runs + 1):
        with open(f'../{algo}/train/scenario_{dataSetName}/{seedOfRun}_meng_individual_{dataSetName}.pkl',
                  "rb") as fileName_individual:
            dict_best_MTGP_individuals = pickle.load(fileName_individual)

        individual = dict_best_MTGP_individuals.get(len(dict_best_MTGP_individuals) - 1)
        replenishment_rule_tree = individual[0]
        transshipment_rule_tree = individual[1]

        replenishment_counts = count_node_appearances(candidate_terminals_replenishment, replenishment_rule_tree)
        transshipment_counts = count_node_appearances(candidate_terminals_transshipment, transshipment_rule_tree)

        for terminal in candidate_terminals_replenishment:
            replenishment_total_counts[terminal] += replenishment_counts[terminal]

        for terminal in candidate_terminals_transshipment:
            transshipment_total_counts[terminal] += transshipment_counts[terminal]

    replenishment_mean_counts = {terminal: count / num_runs for terminal, count in replenishment_total_counts.items()}
    transshipment_mean_counts = {terminal: count / num_runs for terminal, count in transshipment_total_counts.items()}

    return replenishment_mean_counts, transshipment_mean_counts


def plot_scenarios_replenishment(algo, scenarios):
    num_scenarios = len(scenarios)
    cols = 1  # Two columns: one for replenishment and one for transshipment
    rows = num_scenarios  # Calculate number of rows needed

    # Increase the subplot height for better visibility
    subplot_height = 300  # Height of each subplot

    # Only for replenishment
    sub_plot_replenishment = [f'Replenishment\n{scenario}' for scenario in scenarios]
    sub_plot_transshipment = [f'Transshipment\n{scenario}' for scenario in scenarios]
    sub_plot_titles = []
    for i in range(len(sub_plot_replenishment)):
        sub_plot_titles.append(sub_plot_replenishment[i])
        # sub_plot_titles.append(sub_plot_transshipment[i])

        # Define colors for specific nodes
        color_map = {
            '+': 'orange',
            '-': 'orange',
            '*': 'orange',
            '/': 'orange',
            'max': 'orange',
            'min': 'orange',
            'sqrt': 'orange',
            '^2': 'orange'
        }

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=sub_plot_titles,
        shared_yaxes=False,
        vertical_spacing=0.13,  # Reduced spacing to fit larger subplots
        horizontal_spacing=0.05
    )

    # Update font size of subplot titles through layout annotations
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=18)  # Set title font size to 18 (adjust as needed)

    for i, scenario in enumerate(scenarios):
        replenishment_counts, transshipment_counts = main(algo, scenario)
        print("Scenario: " + str(i))
        print("Replenishment: " + str(replenishment_counts))
        print("Transshipment: " + str(transshipment_counts))

        row = i + 1
        col_replenishment = 1
        # col_transshipment = 2

        # Add bar chart for replenishment
        fig.add_trace(
            go.Bar(
                x=list(replenishment_counts.keys()),
                y=list(replenishment_counts.values()),
                name=f'Replenishment - {scenario}',
                marker_color=[color_map.get(node, 'blue') for node in replenishment_counts.keys()]
                # marker_color='blue'
            ),
            row=row, col=col_replenishment
        )

        # Add bar chart for transshipment
        # fig.add_trace(
        #     go.Bar(
        #         x=list(transshipment_counts.keys()),
        #         y=list(transshipment_counts.values()),
        #         name=f'Transshipment - {scenario}',
        #         marker_color=[color_map.get(node, 'green') for node in transshipment_counts.keys()]
        #         # marker_color='green'
        #     ),
        #     row=row, col=col_transshipment
        # )

    fig.update_layout(
        height=subplot_height * rows,  # Adjust height based on the number of rows
        width=800,
        # title_text='Mean Node Counts Across Scenarios',
        barmode='group',
        showlegend=False,  # Hide legend to prevent clutter
        font=dict(size=18),
        # xaxis_title='Node',
        # yaxis_title='Mean Count'
    )

    plot_dir = f'../{algo}/train/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    plot_file_path = os.path.join(plot_dir, 'mean_node_counts_replenishment.png')
    fig.write_image(plot_file_path)
    # Save the figure as a PDF

    fig.show()

def plot_scenarios_transshipment(algo, scenarios):
    num_scenarios = len(scenarios)
    cols = 1  # Two columns: one for replenishment and one for transshipment
    rows = num_scenarios  # Calculate number of rows needed

    # Increase the subplot height for better visibility
    subplot_height = 300  # Height of each subplot

    # Only for replenishment
    sub_plot_replenishment = [f'Replenishment\n{scenario}' for scenario in scenarios]
    sub_plot_transshipment = [f'Transshipment\n{scenario}' for scenario in scenarios]
    sub_plot_titles = []
    for i in range(len(sub_plot_replenishment)):
        # sub_plot_titles.append(sub_plot_replenishment[i])
        sub_plot_titles.append(sub_plot_transshipment[i])

        # Define colors for specific nodes
        color_map = {
            '+': 'orange',
            '-': 'orange',
            '*': 'orange',
            '/': 'orange',
            'max': 'orange',
            'min': 'orange',
            'sqrt': 'orange',
            '^2': 'orange'
        }

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=sub_plot_titles,
        shared_yaxes=False,
        vertical_spacing=0.13,  # Reduced spacing to fit larger subplots
        horizontal_spacing=0.05
    )

    # Update font size of subplot titles through layout annotations
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=18)  # Set title font size to 18 (adjust as needed)

    for i, scenario in enumerate(scenarios):
        replenishment_counts, transshipment_counts = main(algo, scenario)

        row = i + 1
        # col_replenishment = 1
        col_transshipment = 1

        # Add bar chart for replenishment
        # fig.add_trace(
        #     go.Bar(
        #         x=list(replenishment_counts.keys()),
        #         y=list(replenishment_counts.values()),
        #         name=f'Replenishment - {scenario}',
        #         marker_color=[color_map.get(node, 'blue') for node in replenishment_counts.keys()]
        #         # marker_color='blue'
        #     ),
        #     row=row, col=col_replenishment
        # )

        # Add bar chart for transshipment
        fig.add_trace(
            go.Bar(
                x=list(transshipment_counts.keys()),
                y=list(transshipment_counts.values()),
                name=f'Transshipment - {scenario}',
                marker_color=[color_map.get(node, 'green') for node in transshipment_counts.keys()]
                # marker_color='green'
            ),
            row=row, col=col_transshipment
        )

    fig.update_layout(
        height=subplot_height * rows,  # Adjust height based on the number of rows
        width=800,
        # title_text='Mean Node Counts Across Scenarios',
        barmode='group',
        showlegend=False,  # Hide legend to prevent clutter
        font=dict(size=18),
        # xaxis_title='Node',
        # yaxis_title='Mean Count'
    )

    plot_dir = f'../{algo}/train/plots/'
    os.makedirs(plot_dir, exist_ok=True)
    plot_file_path = os.path.join(plot_dir, 'mean_node_counts_transshipment.png')
    fig.write_image(plot_file_path)
    # Save the figure as a PDF

    fig.show()

if __name__ == "__main__":
    algo = 'MTGP_niching'
    scenarios = ["sN2h_1_10b3","mN2h_1_10b3","lN2h_1_10b3"]
    # scenarios = ["lN2h_1_5b2", "lN2h_1_10b3", "lN2h_5_10b5", "lN2h_5_50b10",
    #              "lN2h_10_50b2", "lN2h_10_100b3", "lN2h_50_100b5", "lN2h_100_100b10",
    #              "lN3h_1_5_10b2", "lN3h_1_5_50b3", "lN3h_5_10_50b5", "lN3h_5_5_50b10",
    #              "lN3h_10_50_50b2", "lN3h_10_50_100b3", "lN3h_50_50_50b5", "lN3h_50_50_100b10"]

    plot_scenarios_replenishment(algo, scenarios)
    # plot_scenarios_transshipment(algo, scenarios)
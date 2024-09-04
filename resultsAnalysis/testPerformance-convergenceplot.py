import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
algorithms = ["CCGP", "MTGP", "NichMTGP"]
colors = {
    "CCGP": "blue",
    "NichCCGP": "orange",
    "MTGP": "green",
    "NichMTGP": "red"
}
workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/Results/"
folders = {algo: os.path.join(workdir, algo, "train") for algo in algorithms}

# List of scenarios
scenarios = ["sN2h_1_5b2", "sN2h_1_10b3", "sN2h_5_10b5", "sN2h_5_50b10",
             "sN2h_10_50b2", "sN2h_10_100b3", "sN2h_50_100b5", "sN2h_100_100b10",
             "sN3h_1_5_10b2", "sN3h_1_5_50b3", "sN3h_5_10_50b5", "sN3h_5_5_50b10",
             "sN3h_10_50_50b2", "sN3h_10_50_100b3", "sN3h_50_50_50b5", "sN3h_50_50_100b10"]

runs = 30

# Initialize a dictionary to hold data for all algorithms and scenarios
data = {scenario: [] for scenario in scenarios}

# Read the CSV files and store the data
for algo, folder in folders.items():
    for scenario in scenarios:
        for run in range(1, runs + 1):
            file_path = os.path.join(folder, f'scenario_{scenario}/test/{run}_{scenario}_testResults.csv')
            df = pd.read_csv(file_path)

            gen = len(df['TestFitness'])
            for i in range(gen):
                result = df['TestFitness'][i]
                data[scenario].append({'Algorithm': algo, 'Run': run, 'Gen': i, 'TestFitness': result})

for scenario in data:
    test_fitness_values = [entry['TestFitness'] for entry in data[scenario]]
    worst_value = max(test_fitness_values) if len(test_fitness_values) > 0 else np.nan

    for entry in data[scenario]:
        if np.isnan(entry['TestFitness']):
            entry['TestFitness'] = worst_value

    test_fitness_values = [entry['TestFitness'] for entry in data[scenario]]
    median_value = np.median(test_fitness_values)

    for entry in data[scenario]:
        if entry['TestFitness'] > median_value * 1.5:
            entry['TestFitness'] = median_value * 1.5

# Create subplots for each scenario with reduced spacing
fig = make_subplots(
    rows=4, cols=4, subplot_titles=scenarios, shared_yaxes=False,
    horizontal_spacing=0.05,  # Reduced horizontal spacing
    vertical_spacing=0.05      # Reduced vertical spacing
)

# Add convergence curves for each scenario
for i, scenario in enumerate(scenarios):
    row = i // 4 + 1
    col = i % 4 + 1
    scenario_data = pd.DataFrame(data[scenario])

    # Calculate mean TestFitness across all runs for each generation
    mean_test_fitness = scenario_data.groupby(['Gen', 'Algorithm'])['TestFitness'].mean().reset_index()

    for algo in algorithms:
        algo_data = mean_test_fitness[mean_test_fitness['Algorithm'] == algo]
        fig.add_trace(
            go.Scatter(
                x=algo_data['Gen'],
                y=algo_data['TestFitness'],
                mode='lines',
                name=algo,
                line=dict(color=colors[algo]),
                showlegend=(i == 0)  # Show legend only once (in the first subplot)
            ),
            row=row, col=col
        )

# Update overall layout to place the legend at the bottom
fig.update_layout(
    height=1200, width=1200,
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(size=13),
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.03,  # Place legend below the plot
        xanchor="center",
        yanchor="top"
    )
)

# Save the figure as a PDF
fig.write_image(workdir + "convergence_curves.pdf")

# Show the figure
fig.show()
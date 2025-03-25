import math
import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
algorithms = ["GP", "redGPd", "redGPa"]
algo_legend = ["GP", "redGPd-", "redGP"]
colors = {
    "GP": "blue",
    "redGPd": "orange",
    "redGPa": "green",
    "redGP": "red",
    "redGPd-": "orange",
    "redGPa-": "green",
}
workdir = "C:/Users/I3Nexus/Desktop/2025 paper/Paper1-AdaptiveContinuousSearchSpaceReduction/Results/"
folders = {algo: os.path.join(workdir, algo) for algo in algorithms}

# List of scenarios
scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_1_5b5"]
# scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_1_5b5",
#              "sN2h_1_10b2", "sN2h_1_10b3", "sN2h_1_10b5",
#              "sN2h_5_10b2", "sN2h_5_10b3", "sN2h_5_10b5",
#              "lN3h_1_5_10b2", "lN3h_1_5_10b3", "lN3h_1_5_10b5",
#              "lN3h_1_1_10b2", "lN3h_1_1_10b3", "lN3h_1_1_10b5",
#              "lN3h_5_10_5b2", "lN3h_5_10_5b3", "lN3h_5_10_5b5"]

runs = 30

# Initialize a dictionary to hold data for all algorithms and scenarios
data = {scenario: [] for scenario in scenarios}

# Read the CSV files and store the data
for algo, folder in folders.items():
    if algo == "GP":
        algo_name = "GP"
    elif algo == "redGPd":
        algo_name = "redGPd-"
    elif algo == "redGPa":
        algo_name = "redGP"
    for scenario in scenarios:
        for run in range(0, runs):
            file_path = os.path.join(folder, f'scenario_{scenario}/test/{run}_{scenario}_testResults.csv')
            df = pd.read_csv(file_path)

            gen = len(df['PCDiversity'])
            for i in range(gen):
                result = df['PCDiversity'][i]
                data[scenario].append({'Algorithm': algo_name, 'Run': run, 'Gen': i, 'PCDiversity': result})

for scenario in data:
    test_fitness_values = [entry['PCDiversity'] for entry in data[scenario]]
    worst_value = max(test_fitness_values) if len(test_fitness_values) > 0 else np.nan

    for entry in data[scenario]:
        if np.isnan(entry['PCDiversity']):
            entry['PCDiversity'] = worst_value

    test_fitness_values = [entry['PCDiversity'] for entry in data[scenario]]
    median_value = np.median(test_fitness_values)


# Create subplots for each scenario with reduced spacing
# fig = make_subplots(
#     rows=4, cols=4, subplot_titles=scenarios, shared_yaxes=False,
#     horizontal_spacing=0.05,  # Reduced horizontal spacing
#     vertical_spacing=0.05      # Reduced vertical spacing
# )
# Calculate number of rows and columns
num_scenarios = len(scenarios)
row = math.ceil(num_scenarios / 5)  # Ceiling to ensure enough rows for all scenarios
col = min(num_scenarios, 5)  # Max of 4 columns or fewer for the last row
width_each = 250
height_each = 265

fig = make_subplots(
    rows=1, cols=3, subplot_titles=scenarios, shared_yaxes=False,
    horizontal_spacing=0.05,  # Reduced horizontal spacing
    vertical_spacing=0.05      # Reduced vertical spacing
)

# Add convergence curves for each scenario
for i, scenario in enumerate(scenarios):
    row = i // 3 + 1
    col = i % 3 + 1
    scenario_data = pd.DataFrame(data[scenario])

    # Calculate mean TestFitness across all runs for each generation
    mean_test_fitness = scenario_data.groupby(['Gen', 'Algorithm'])['PCDiversity'].mean().reset_index()

    for algo in algo_legend:
        algo_data = mean_test_fitness[mean_test_fitness['Algorithm'] == algo]
        fig.add_trace(
            go.Scatter(
                x=algo_data['Gen'],
                y=algo_data['PCDiversity'],
                mode='lines',
                name=algo,
                line=dict(color=colors[algo]),
                showlegend=(i == 0)  # Show legend only once (in the first subplot)
            ),
            row=row, col=col
        )

# Update overall layout to place the legend at the bottom
fig.update_layout(
    height=265, width=750,
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(size=13),
    legend=dict(
        orientation="h",
        x=0.5, y=-0.08,
        xanchor="center", yanchor="top"
    )
)

# Save the figure as a PDF
fig.write_image(workdir + "PCDiversity_convergence_curves.png", engine="kaleido")

# Show the figure
fig.show()
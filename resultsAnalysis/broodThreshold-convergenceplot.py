import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
algorithms = ["GP", "redGPa", "redGPd", "redGP"]
colors = {"GP": "blue", "redGPa": "orange", "redGPd": "green", "redGP": "red"}

# Paths
workdir = "C:/Users/I3Nexus/Desktop/2025 paper/Paper1-AdaptiveContinuousSearchSpaceReduction/Results/"
folders = {algo: os.path.join(workdir, algo) for algo in algorithms}

# Scenarios and runs
scenarios = [
    "sN2h_1_5b2", "sN2h_1_5b3", "sN2h_1_5b5",
    "sN2h_1_10b2", "sN2h_1_10b3", "sN2h_1_10b5",
    "sN2h_5_10b2", "sN2h_5_10b3", "sN2h_5_10b5",
    "lN3h_1_5_10b2", "lN3h_1_5_10b3", "lN3h_1_5_10b5",
    "lN3h_1_1_10b2", "lN3h_1_1_10b3", "lN3h_1_1_10b5",
    "lN3h_5_10_5b2", "lN3h_5_10_5b3", "lN3h_5_10_5b5"
]
runs = 30

# Data collection
data = {scenario: [] for scenario in scenarios}

for algo, folder in folders.items():
    for scenario in scenarios:
        for run in range(runs):
            file_path = os.path.join(folder, f'scenario_{scenario}/{run}_{scenario}_BroodThreshold.csv')
            df = pd.read_csv(file_path)

            gen = len(df['BroodThreshold'])
            for i in range(gen):
                result = df['BroodThreshold'][i]
                data[scenario].append({'Algorithm': algo, 'Run': run, 'Gen': i, 'BroodThreshold': result})

# Plot convergence curves
fig = make_subplots(
    rows=5, cols=5, subplot_titles=scenarios, shared_yaxes=False,
    horizontal_spacing=0.05, vertical_spacing=0.05
)

for i, scenario in enumerate(scenarios):
    row = i // 5 + 1
    col = i % 5 + 1
    scenario_data = pd.DataFrame(data[scenario])

    # Calculate mean and std for each algorithm per generation
    mean_test_fitness = scenario_data.groupby(['Gen', 'Algorithm'])['BroodThreshold'].mean().reset_index()
    std_test_fitness = scenario_data.groupby(['Gen', 'Algorithm'])['BroodThreshold'].std().reset_index()
    merged_fitness = pd.merge(mean_test_fitness, std_test_fitness, on=['Gen', 'Algorithm'], suffixes=('_mean', '_std'))

    print(f"\n--- Scenario: {scenario} ---")
    for algo in algorithms:
        algo_data = merged_fitness[merged_fitness['Algorithm'] == algo]
        mean_val = algo_data['BroodThreshold_mean'].mean()
        std_val = algo_data['BroodThreshold_std'].mean()
        print(f"{algo}: {mean_val:.2f} ({std_val:.2f})")

        # Add traces for visualization
        fig.add_trace(
            go.Scatter(
                x=algo_data['Gen'],
                y=algo_data['BroodThreshold_mean'],
                mode='lines',
                name=algo if i == 0 else None,
                line=dict(color=colors[algo]),
                showlegend=(i == 0)  # Show legend only once (in the first subplot)
            ),
            row=row, col=col
        )

# Layout settings
fig.update_layout(
    height=1200, width=1200,
    margin=dict(l=20, r=20, t=20, b=20),
    font=dict(size=13),
    legend=dict(
        orientation="h",
        x=0.5, y=-0.03,
        xanchor="center", yanchor="top"
    )
)

# Save and show
fig.write_image(workdir + "BroodThreshold_convergence_curves_redGP.pdf")
fig.show()

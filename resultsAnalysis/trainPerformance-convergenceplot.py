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

# Paths
workdir = "C:/Users/I3Nexus/Desktop/2025 paper/Paper1-AdaptiveContinuousSearchSpaceReduction/Results/"
folders = {algo: os.path.join(workdir, algo) for algo in algorithms}

# Scenarios and runs
# List of scenarios
scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_1_5b5"]
# scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_1_5b5",
#              "sN2h_1_10b2", "sN2h_1_10b3", "sN2h_1_10b5",
#              "sN2h_5_10b2", "sN2h_5_10b3", "sN2h_5_10b5"]
scenarios_type = 'small'
# scenarios = ["lN3h_1_5_10b2", "lN3h_1_5_10b3", "lN3h_1_5_10b5",
#              "lN3h_1_1_10b2", "lN3h_1_1_10b3", "lN3h_1_1_10b5",
#              "lN3h_5_10_5b2", "lN3h_5_10_5b3", "lN3h_5_10_5b5"]
# scenarios_type = 'large'
runs = 30

# Data collection
data = {scenario: [] for scenario in scenarios}

for algo, folder in folders.items():
    if algo == "GP":
        algo_name = "GP"
    elif algo == "redGPd":
        algo_name = "redGPd-"
    elif algo == "redGPa":
        algo_name = "redGP"
    for scenario in scenarios:
        for run in range(runs):
            file_path = os.path.join(folder, f'scenario_{scenario}/{run}_min_fitness{scenario}.npy')
            if os.path.exists(file_path):
                fitness_values = np.load(file_path)
                for i, value in enumerate(fitness_values):
                    data[scenario].append({'Algorithm': algo_name, 'Run': run, 'Gen': i, 'TrainFitness': value})

# Plot convergence curves
fig = make_subplots(
    rows=1, cols=3, subplot_titles=scenarios, shared_yaxes=False,
    horizontal_spacing=0.05, vertical_spacing=0.05
)

for i, scenario in enumerate(scenarios):
    row = i // 3 + 1
    col = i % 3 + 1
    scenario_data = pd.DataFrame(data[scenario])

    # Calculate mean and std for each algorithm per generation
    mean_test_fitness = scenario_data.groupby(['Gen', 'Algorithm'])['TrainFitness'].mean().reset_index()
    std_test_fitness = scenario_data.groupby(['Gen', 'Algorithm'])['TrainFitness'].std().reset_index()
    merged_fitness = pd.merge(mean_test_fitness, std_test_fitness, on=['Gen', 'Algorithm'], suffixes=('_mean', '_std'))

    printContent = f""
    print(f"\n--- Scenario: {scenario} ---")
    for algo in algo_legend:
        algo_data = merged_fitness[merged_fitness['Algorithm'] == algo]
        mean_val = algo_data['TrainFitness_mean'].mean()
        std_val = algo_data['TrainFitness_std'].mean()
        # print(f"{algo}: {mean_val:.2f} ({std_val:.2f})")
        printContent += f" & {mean_val:.2f}({std_val:.2f})"

        # Add traces for visualization
        fig.add_trace(
            go.Scatter(
                x=algo_data['Gen'],
                y=algo_data['TrainFitness_mean'],
                mode='lines',
                name=algo if i == 0 else None,
                line=dict(color=colors[algo]),
                showlegend=(i == 0)  # Show legend only once (in the first subplot)
            ),
            row=row, col=col
        )
    printContent += r"\\"
    print(printContent)

# Layout settings
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

# Save and show
fig.write_image(workdir + "training_convergence_curves_redGP.png")
# fig.write_image(workdir + "training_convergence_curves_redGP.pdf")
fig.show()

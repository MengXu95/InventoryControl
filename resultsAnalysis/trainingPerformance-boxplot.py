import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
algorithms = ["PSO", "CCGP", "MTGP"]
colors = {"PSO": "blue", "CCGP": "orange", "MTGP": "green"}

# Define working directory
workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/Results/"
folders = {algo: os.path.join(workdir, algo, "train") for algo in algorithms}

# List of small scenarios
scenarios = ["sN2h_1_5b2"]
# runs = [0, 1, 2, 4, 6]
runs = [0, 1, 2, 3, 4, 5, 6]

data = {algo: [] for algo in algorithms}  # Stores all final generation values

# Read the CSV files and store the final generation data
for algo in algorithms:
    for scenario in scenarios:
        for run in runs:
            if algo == 'MTGP':
                directory = f'../MTGP_niching_rental/train/scenario_{scenario}-400popsize/'
            elif algo == 'CCGP':
                directory = f'../CCGP_niching_rental/train/scenario_{scenario}-400popsize/'
            elif algo == 'PSO':
                directory = f'../PSO_rental/train/scenario_{scenario}-400popsize/'

            file_path = os.path.join(directory, f'{run}_min_fitness{scenario}.npy')
            fitness_values = np.load(file_path)  # Load fitness values
            data[algo].append(fitness_values[-1])  # Store only the final generation value

# Create box plot for final generation
fig = go.Figure()

for algo in algorithms:
    fig.add_trace(go.Box(
        y=data[algo],
        name=algo,
        marker=dict(color=colors[algo])
    ))

fig.update_layout(
    title="Box Plot of Final Generation Fitness Values",
    yaxis_title="Final Total Cost",
    template="plotly_white",
    width=600,
    height=600,
    showlegend=False
)

fig.show()

# Print mean and std of final generation
for algo in algorithms:
    final_gen_values = np.array(data[algo])
    print(f"{algo}: Mean = {np.mean(final_gen_values):.4f}, Std = {np.std(final_gen_values):.4f}")

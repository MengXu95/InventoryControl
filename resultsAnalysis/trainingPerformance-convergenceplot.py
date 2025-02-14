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
runs = 3

data = {algo: [] for algo in algorithms}  # Stores all runs for averaging

# Read the CSV files and store the data
for algo in algorithms:
    for scenario in scenarios:
        algo_data = []  # Store all runs for this algorithm
        for run in [0,1,2,4,6]:
            if algo == 'MTGP':
                directory = f'../MTGP_niching_rental/train/scenario_{scenario}-400popsize/'
            elif algo == 'CCGP':
                directory = f'../CCGP_niching_rental/train/scenario_{scenario}-400popsize/'
            elif algo == 'PSO':
                directory = f'../PSO_rental/train/scenario_{scenario}-400popsize/'

            file_path = os.path.join(directory, f'{run}_min_fitness{scenario}.npy')
            fitness_values = np.load(file_path)  # Load fitness values
            algo_data.append(fitness_values)

        data[algo] = np.array(algo_data)  # Convert to numpy array



# Plot convergence curves
fig = make_subplots(rows=1, cols=1, subplot_titles=["Convergence Curves"])

# Define figure size
fig_width = 600  # Set width in pixels
fig_height = 600  # Set height in pixels

generations = np.arange(data["PSO"].shape[1])  # Assume all algorithms have the same number of generations

for algo in algorithms:
    mean_fitness = np.mean(data[algo], axis=0)
    std_fitness = np.std(data[algo], axis=0)

    fig.add_trace(go.Scatter(
        x=generations, y=mean_fitness,
        mode='lines', name=f'{algo} Mean',
        line=dict(color=colors[algo])
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([generations, generations[::-1]]),
        y=np.concatenate([mean_fitness - std_fitness, (mean_fitness + std_fitness)[::-1]]),
        fill='toself', fillcolor=colors[algo], opacity=0.3,
        line=dict(color='rgba(255,255,255,0)'), name=f'{algo} Std Dev'
    ))

fig.update_layout(
    title="Convergence Curves",
    xaxis_title="Generations",
    yaxis_title="Total cost",
    template="plotly_white",
    width=fig_width,
    height=fig_height,
    legend=dict(
        x=0.75,  # Adjust the position of the legend inside the figure
        y=0.95,
        bgcolor='rgba(255,255,255,0.5)',
        bordercolor='black',
        borderwidth=1
    )
)

# fig.update_layout(title="Convergence Curves", xaxis_title="Generations", yaxis_title="Total cost", template="plotly_white")
fig.show()

# Print mean and std of final generation
for algo in algorithms:
    final_gen_values = data[algo][:, -1]  # Extract last generation values
    # print(final_gen_values)
    print(f"{algo}: {np.mean(final_gen_values):.4f} ({np.std(final_gen_values):.4f})")

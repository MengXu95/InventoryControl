import os
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
# algorithms = ["PSO", "CCGP", "MTGP"]
algorithms = ["CCGP", "MTGP"]
colors = {"PSO": "blue", "CCGP": "orange", "MTGP": "green"}

# Define working directory
workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/Results/"
folders = {algo: os.path.join(workdir, algo, "train") for algo in algorithms}

# List of small scenarios
scenarios = ["sN2h_1_5b2"]
runs = 2


data = {algo: [] for algo in algorithms}  # Stores all runs for averaging
data_lost_sale = {algo: [] for algo in algorithms}
data_order = {algo: [] for algo in algorithms}
data_rental = {algo: [] for algo in algorithms}

# Read the CSV files and store the data
for algo in algorithms:
    for scenario in scenarios:
        algo_data = []
        algo_data_lost_sale = []  # Store all runs for this algorithm
        algo_data_order = []
        algo_data_rental = []
        for run in range(runs):
            if algo == 'MTGP':
                directory = f'../MTGP_niching_rental/train/scenario_{scenario}/'
            elif algo == 'CCGP':
                directory = f'../CCGP_niching_rental/train/scenario_{scenario}/'
            elif algo == 'PSO':
                directory = f'../PSO_rental/train/scenario_{scenario}/'

            file_path = os.path.join(directory, f'{run}_min_all_cost{scenario}.npy')
            fitness_values = np.load(file_path)  # Load fitness values
            fitness_total_cost = []
            fitness_lost_sale_cost = []
            fitness_order_cost = []
            fitness_rental_cost = []

            for tran_cost, lost_sale_cost, order_cost, rental_cost in fitness_values:
                fitness_total_cost.append(tran_cost+lost_sale_cost+order_cost+rental_cost)
                fitness_lost_sale_cost.append(lost_sale_cost)
                fitness_order_cost.append(order_cost)
                fitness_rental_cost.append(rental_cost)

            algo_data.append(fitness_total_cost)
            algo_data_lost_sale.append(fitness_lost_sale_cost)
            algo_data_order.append(fitness_order_cost)
            algo_data_rental.append(fitness_rental_cost)

        data[algo] = np.array(algo_data)
        data_lost_sale[algo] = np.array(algo_data_lost_sale)  # Convert to numpy array
        data_order[algo] = np.array(algo_data_order)
        data_rental[algo] = np.array(algo_data_rental)

# Plot convergence curves for total cost -----------------------------------
fig = make_subplots(rows=1, cols=1, subplot_titles=["Convergence Curves"])

# Define figure size
fig_width = 600  # Set width in pixels
fig_height = 600  # Set height in pixels

generations = np.arange(data["MTGP"].shape[1])  # Assume all algorithms have the same number of generations

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
print("\nTotal cost: ")
for algo in algorithms:
    final_gen_values = data[algo][:, -1]  # Extract last generation values
    print(f"{algo}: {np.mean(final_gen_values):.4f} ({np.std(final_gen_values):.4f})")


# Plot convergence curves for lost sale cost -----------------------------------
fig = make_subplots(rows=1, cols=1, subplot_titles=["Convergence Curves"])

# Define figure size
fig_width = 600  # Set width in pixels
fig_height = 600  # Set height in pixels

generations = np.arange(data_lost_sale["MTGP"].shape[1])  # Assume all algorithms have the same number of generations

for algo in algorithms:
    mean_fitness = np.mean(data_lost_sale[algo], axis=0)
    std_fitness = np.std(data_lost_sale[algo], axis=0)

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
    yaxis_title="Lost sale cost",
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
print("\nLost sale cost: ")
for algo in algorithms:
    final_gen_values = data_lost_sale[algo][:, -1]  # Extract last generation values
    print(f"{algo}: {np.mean(final_gen_values):.4f} ({np.std(final_gen_values):.4f})")


# Plot convergence curves for order cost -----------------------------------
fig = make_subplots(rows=1, cols=1, subplot_titles=["Convergence Curves"])

# Define figure size
fig_width = 600  # Set width in pixels
fig_height = 600  # Set height in pixels

generations = np.arange(data_order["MTGP"].shape[1])  # Assume all algorithms have the same number of generations

for algo in algorithms:
    mean_fitness = np.mean(data_order[algo], axis=0)
    std_fitness = np.std(data_order[algo], axis=0)

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
    yaxis_title="Order cost",
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
print("\nOrder cost: ")
for algo in algorithms:
    final_gen_values = data_order[algo][:, -1]  # Extract last generation values
    print(f"{algo}: {np.mean(final_gen_values):.4f} ({np.std(final_gen_values):.4f})")


# Plot convergence curves for rental cost -----------------------------------
fig = make_subplots(rows=1, cols=1, subplot_titles=["Convergence Curves"])

# Define figure size
fig_width = 600  # Set width in pixels
fig_height = 600  # Set height in pixels

generations = np.arange(data_rental["MTGP"].shape[1])  # Assume all algorithms have the same number of generations

for algo in algorithms:
    mean_fitness = np.mean(data_rental[algo], axis=0)
    std_fitness = np.std(data_rental[algo], axis=0)

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
    yaxis_title="Rental cost",
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
print("\nRental cost: ")
for algo in algorithms:
    final_gen_values = data_rental[algo][:, -1]  # Extract last generation values
    print(f"{algo}: {np.mean(final_gen_values):.4f} ({np.std(final_gen_values):.4f})")




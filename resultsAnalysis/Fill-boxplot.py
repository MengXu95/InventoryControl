import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
algorithms = ["DRL", "NichMTGP"]
colors = {
    "CCGP": "blue",
    "DRL": "pink",
    "sSPolicy": "orange",
    "MTGP": "green",
    "NichMTGP": "red"
}
workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/Results/"
folders = {algo: os.path.join(workdir, algo, "train") for algo in algorithms}

# List of small scenarios
scenarios = ["sN2h_1_5b2"]
scenarios_type = 'small'
# List of medium scenarios
# scenarios = ["mN2h_1_5b2", "mN2h_1_10b3", "mN2h_5_10b5", "mN2h_5_50b10",
#              "mN2h_10_50b2", "mN2h_10_100b3", "mN2h_50_100b5", "mN2h_100_100b10",
#              "mN3h_1_5_10b2", "mN3h_1_5_50b3", "mN3h_5_10_50b5", "mN3h_5_5_50b10",
#              "mN3h_10_50_50b2", "mN3h_10_50_100b3", "mN3h_50_50_50b5", "mN3h_50_50_100b10"]
# scenarios_type = 'medium'
# List of large scenarios
# scenarios = ["lN2h_1_5b2", "lN2h_1_10b3", "lN2h_5_10b5", "lN2h_5_50b10",
#              "lN2h_10_50b2", "lN2h_10_100b3", "lN2h_50_100b5", "lN2h_100_100b10",
#              "lN3h_1_5_10b2", "lN3h_1_5_50b3", "lN3h_5_10_50b5", "lN3h_5_5_50b10",
#              "lN3h_10_50_50b2", "lN3h_10_50_100b3", "lN3h_50_50_50b5", "lN3h_50_50_100b10"]
# scenarios_type = 'large'

runs = 30

# Initialize a dictionary to hold data for all algorithms and scenarios
data = {scenario: [] for scenario in scenarios}

# Read the CSV files and store the data
for algo, folder in folders.items():
    for scenario in scenarios:
        for run in range(1, runs + 1):
            if algo == 'sSPolicy':
                file_path = os.path.join(folder, f'scenario_{scenario}/{run}_{scenario}_sSPolicy_test_results.csv')
                df = pd.read_csv(file_path)
                result = df['Fill'].iloc[0]
                data[scenario].append({'Algorithm': algo, 'Run': run, 'Fill': result})
            else:
                file_path = os.path.join(folder, f'scenario_{scenario}/test/{run}_{scenario}_testResults.csv')
                df = pd.read_csv(file_path)
                gen = len(df['Fill'])
                result = df['Fill'][gen - 1]
                data[scenario].append({'Algorithm': algo, 'Run': run, 'Fill': result})


for scenario in data:
    test_fitness_values = [entry['Fill'] for entry in data[scenario]]
    worst_value = max(test_fitness_values) if len(test_fitness_values) > 0 else np.nan

    for entry in data[scenario]:
        if np.isnan(entry['Fill']):
            entry['Fill'] = worst_value

    test_fitness_values = [entry['Fill'] for entry in data[scenario]]
    median_value = np.median(test_fitness_values)

    # for entry in data[scenario]:
    #     if entry['TestFitness'] > median_value * 1.5:
    #         entry['TestFitness'] = median_value * 1.5

# Create subplots for each scenario with reduced spacing
fig = make_subplots(
    rows=4, cols=4, subplot_titles=scenarios, shared_yaxes=False,
    horizontal_spacing=0.05,  # Reduced horizontal spacing
    vertical_spacing=0.08      # Reduced vertical spacing
)

# Add boxplots for each scenario
for i, scenario in enumerate(scenarios):
    row = i // 4 + 1
    col = i % 4 + 1
    scenario_data = pd.DataFrame(data[scenario])

    # Calculate y-axis range for this scenario
    y_min = scenario_data['Fill'].min()
    y_max = scenario_data['Fill'].max()

    for algo in algorithms:
        algo_data = scenario_data[scenario_data['Algorithm'] == algo]
        fig.add_trace(
            go.Box(
                y=algo_data['Fill'],
                name=algo,
                marker_color=colors[algo],
                boxmean=True
            ),
            # go.Violin(
            #     y=algo_data['TestFitness'],
            #     name=algo,
            #     marker_color=colors[algo],
            #     box_visible=True,  # Show box inside the violin plot
            #     meanline_visible=True
            # ),
            row=row, col=col
        )

    # Update y-axis range for this subplot
    fig.update_yaxes(range=[y_min, y_max], row=row, col=col)

# Update layout to improve overall appearance
fig.update_layout(
    height=1200, width=1200,
    # title_text="Test Fitness Comparison across Scenarios",
    showlegend=False,
    margin=dict(l=20, r=20, t=30, b=10),  # Adjust margins to make the figure more compact
    font=dict(size=13)  # Adjust font size to fit the subplots better
)

# Save the figure as a PDF
fig.write_image(workdir + "test_Fill_comparison_" + scenarios_type + ".pdf")

# Show the figure
fig.show()



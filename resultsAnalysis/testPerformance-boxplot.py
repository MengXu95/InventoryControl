import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Algorithms and their associated colors
algorithms = ["CCGP", "NichCCGP", "MTGP", "NichMTGP"]
colors = {
    "CCGP": "blue",
    "NichCCGP": "orange",
    "MTGP": "green",
    "NichMTGP": "red"
}
workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/Results/"
folders = {algo: os.path.join(workdir, algo, "train") for algo in algorithms}

# List of scenarios
scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_5_10b5", "sN2h_5_10b10",
             "sN2h_10_50b2", "sN2h_10_50b3", "sN2h_50_100b5", "sN2h_100_100b10",
             "sN3h_5_10_50b5", "sN3h_5_5_50b10", "sN3h_10_50_50b2",
             "sN3h_10_50_100b3", "sN3h_50_50_50b5", "sN3h_50_50_100b10"]

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
            result = df['TestFitness'][gen - 1]
            data[scenario].append({'Algorithm': algo, 'Run': run, 'TestFitness': result})

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

# Add boxplots for each scenario
for i, scenario in enumerate(scenarios):
    row = i // 4 + 1
    col = i % 4 + 1
    scenario_data = pd.DataFrame(data[scenario])

    # Calculate y-axis range for this scenario
    y_min = scenario_data['TestFitness'].min()
    y_max = scenario_data['TestFitness'].max()

    for algo in algorithms:
        algo_data = scenario_data[scenario_data['Algorithm'] == algo]
        fig.add_trace(
            go.Box(
                y=algo_data['TestFitness'],
                name=algo,
                marker_color=colors[algo],
                boxmean=True
            ),
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
fig.write_image(workdir + "test_fitness_comparison.pdf")

# Show the figure
fig.show()



# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# algorithms = ["ResultsCCGP", "ResultsNichCCGP", "ResultsMTGP", "ResultsNichMTGP"]
# workdir = "C:/Users/I3Nexus/Desktop/PaperInventoryManagement/"
# # Set the paths to the folders containing the CSV files for each algorithm
# folders = {}
# for algo in algorithms:
#     path = workdir + algo + "/train/"
#     folders[algo] = path
#
#
# # List of scenarios (assuming they are named consistently across all folders)
# scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_5_10b5", "sN2h_5_10b10",
#              "sN2h_10_50b2", "sN2h_10_50b3", "sN2h_50_100b5", "sN2h_100_100b10",
#              "sN3h_5_10_50b5", "sN3h_5_5_50b10",
#              "sN3h_10_50_100b3", "sN3h_50_50_50b5", "sN3h_50_50_100b10"
#              ]
#
# # scenarios = ["sN2h_1_5b2", "sN2h_1_5b3", "sN2h_5_10b5", "sN2h_5_10b10",
# #              "sN2h_10_50b2", "sN2h_10_50b3", "sN2h_50_100b5", "sN2h_100_100b10",
# #              "sN3h_1_5_10b2", "sN3h_1_5_50b3", "sN3h_5_10_50b5", "sN3h_5_5_50b10",
# #              "sN3h_10_50_50b2", "sN3h_10_50_100b3", "sN3h_50_50_50b5", "sN3h_50_50_100b10"]
# runs = 30
#
# # Initialize a dictionary to hold data for all algorithms and scenarios
# data = {scenario: [] for scenario in scenarios}
#
# # Read the CSV files and store the data
# for algo, folder in folders.items():
#     for scenario in scenarios:
#         for run in range(1,runs+1):
#             file_path = os.path.join(folder, f'scenario_{scenario}/test/{run}_{scenario}_testResults.csv')
#             # print(file_path)
#             df = pd.read_csv(file_path)
#
#             gen = len(df['TestFitness'])
#             # Assume that the result is in a column named 'Result'
#             result = df['TestFitness'][gen-1]
#             data[scenario].append({'Algorithm': algo, 'Run': run, 'TestFitness': result})
#
# for scenario in data:
#     # Extract all TestFitness values for this scenario
#     test_fitness_values = [entry['TestFitness'] for entry in data[scenario]]
#
#     # Determine the worst value (assuming higher values are worse; change as necessary)
#     worst_value = max(test_fitness_values) if len(test_fitness_values) > 0 else np.nan
#
#     # Replace NaN with the worst value
#     for entry in data[scenario]:
#         if np.isnan(entry['TestFitness']):
#             entry['TestFitness'] = worst_value
#
#     test_fitness_values = [entry['TestFitness'] for entry in data[scenario]]
#
#     # Calculate the median value
#     median_value = np.median(test_fitness_values)
#
#     # Replace worst value with the median_value * 3
#     for entry in data[scenario]:
#         if entry['TestFitness'] > median_value * 1.5:
#             entry['TestFitness'] = median_value * 1.5
#
#
# # Convert the data into a pandas DataFrame
# df_all = pd.concat([pd.DataFrame(data[scenario]) for scenario in scenarios])
#
# # Create a boxplot figure with subplots for each scenario
# fig, axes = plt.subplots(4, 4, figsize=(20, 20))
# axes = axes.flatten()
#
# for i, scenario in enumerate(scenarios):
#     sns.boxplot(x='Algorithm', y='TestFitness', data=pd.DataFrame(data[scenario]), ax=axes[i])
#     axes[i].set_title(scenario)
#     axes[i].set_xlabel('')
#     axes[i].set_ylabel('Test Performance')
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

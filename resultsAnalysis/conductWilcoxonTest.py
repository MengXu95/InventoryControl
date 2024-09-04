import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

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

# List of small scenarios
# scenarios = ["sN2h_1_5b2", "sN2h_1_10b3", "sN2h_5_10b5", "sN2h_5_50b10",
#              "sN2h_10_50b2", "sN2h_10_100b3", "sN2h_50_100b5", "sN2h_100_100b10",
#              "sN3h_1_5_10b2", "sN3h_1_5_50b3", "sN3h_5_10_50b5", "sN3h_5_5_50b10",
#              "sN3h_10_50_50b2", "sN3h_10_50_100b3", "sN3h_50_50_50b5", "sN3h_50_50_100b10"]
# scenarios_type = 'small'
# List of medium scenarios
# scenarios = ["mN2h_1_5b2", "mN2h_1_10b3", "mN2h_5_10b5", "mN2h_5_50b10",
#              "mN2h_10_50b2", "mN2h_10_100b3", "mN2h_50_100b5", "mN2h_100_100b10",
#              "mN3h_1_5_10b2", "mN3h_1_5_50b3", "mN3h_5_10_50b5", "mN3h_5_5_50b10",
#              "mN3h_10_50_50b2", "mN3h_10_50_100b3", "mN3h_50_50_50b5", "mN3h_50_50_100b10"]
# scenarios_type = 'medium'
# List of large scenarios
scenarios = ["lN2h_1_5b2", "lN2h_1_10b3", "lN2h_5_10b5", "lN2h_5_50b10",
             "lN2h_10_50b2", "lN2h_10_100b3", "lN2h_50_100b5", "lN2h_100_100b10",
             "lN3h_1_5_10b2", "lN3h_1_5_50b3", "lN3h_5_10_50b5", "lN3h_5_5_50b10",
             "lN3h_10_50_50b2", "lN3h_10_50_100b3", "lN3h_50_50_50b5", "lN3h_50_50_100b10"]
scenarios_type = 'large'

runs = 30
alpha = 0.05  # Significance level

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

    # for entry in data[scenario]:
    #     if entry['TestFitness'] > median_value * 1.5:
    #         entry['TestFitness'] = median_value * 1.5


# Wilcoxon Test
def perform_wilcoxon_test(scenario_data, algorithms):
    results = []
    for i, algo1 in enumerate(algorithms):
        for algo2 in algorithms[i + 1:]:
            data_algo1 = scenario_data[scenario_data['Algorithm'] == algo1]['TestFitness']
            data_algo2 = scenario_data[scenario_data['Algorithm'] == algo2]['TestFitness']

            if len(data_algo1) == len(data_algo2):  # Ensure the data lengths match
                stat, p_value = wilcoxon(data_algo1, data_algo2)
                median_algo1 = data_algo1.median()
                median_algo2 = data_algo2.median()

                if p_value < alpha:
                    # Determine the better algorithm based on the median value
                    if median_algo1 < median_algo2:
                        better_algo = algo1
                    else:
                        better_algo = algo2
                else:
                    better_algo = "No significant difference"

                results.append({
                    'Algorithm 1': algo1,
                    'Algorithm 2': algo2,
                    'p-value': p_value,
                    'Better Algorithm': better_algo,
                    'Median Algo 1': median_algo1,
                    'Median Algo 2': median_algo2
                })
            else:
                results.append({
                    'Algorithm 1': algo1,
                    'Algorithm 2': algo2,
                    'p-value': np.nan,
                    'Better Algorithm': "N/A",
                    'Median Algo 1': np.nan,
                    'Median Algo 2': np.nan
                })
    return results


# Calculate mean and std
def calculate_mean_std(data, algorithms):
    mean_std_results = []
    for scenario in scenarios:
        scenario_data = pd.DataFrame(data[scenario])
        for algo in algorithms:
            algo_data = scenario_data[scenario_data['Algorithm'] == algo]['TestFitness']
            mean_value = algo_data.mean()
            std_value = algo_data.std()
            mean_std_results.append({
                'Scenario': scenario,
                'Algorithm': algo,
                'Mean': mean_value,
                'Standard Deviation': std_value
            })
    return mean_std_results

def calculate_mean_std_easy_for_table(data, algorithms):
    mean_std_results = []
    for scenario in scenarios:
        scenario_data = pd.DataFrame(data[scenario])
        for algo in algorithms:
            algo_data = scenario_data[scenario_data['Algorithm'] == algo]['TestFitness']
            mean_value = algo_data.mean()
            best_value = algo_data.min()
            std_value = algo_data.std()
            mean_std_results.append({
                'Scenario': scenario,
                'Algorithm': algo,
                'Mean': str(round(mean_value,2))+'('+str(round(std_value,2))+')',
                'Best': round(best_value,2)
            })
    return mean_std_results

# Perform Wilcoxon test and calculate mean/std
wilcoxon_results = {}
mean_std_results = calculate_mean_std(data, algorithms)
mean_std_results_easy_for_table = calculate_mean_std_easy_for_table(data, algorithms)

for scenario in scenarios:
    scenario_data = pd.DataFrame(data[scenario])
    results = perform_wilcoxon_test(scenario_data, algorithms)
    wilcoxon_results[scenario] = results

# Save Wilcoxon results to a CSV file
wilcoxon_rows = []
for scenario, results in wilcoxon_results.items():
    for result in results:
        wilcoxon_rows.append([scenario] + list(result.values()))

wilcoxon_results_df = pd.DataFrame(wilcoxon_rows,
                                   columns=['Scenario', 'Algorithm 1', 'Algorithm 2', 'p-value', 'Better Algorithm',
                                            'Median Algo 1', 'Median Algo 2'])
wilcoxon_results_df.to_csv(os.path.join(workdir, "wilcoxon_test_results_" + scenarios_type + ".csv"), index=False)

# Save Mean/Std results to a CSV file
mean_std_df = pd.DataFrame(mean_std_results, columns=['Scenario', 'Algorithm', 'Mean', 'Standard Deviation'])
mean_std_df.to_csv(os.path.join(workdir, "mean_std_results_" + scenarios_type + ".csv"), index=False)

mean_std_df_easy_for_table = pd.DataFrame(mean_std_results_easy_for_table, columns=['Scenario', 'Algorithm', 'Mean', 'Best'])
mean_std_df_easy_for_table.to_csv(os.path.join(workdir, "mean_std_results_easy_for_table_" + scenarios_type + ".csv"), index=False)

# Print the results (optional)
for scenario, results in wilcoxon_results.items():
    print(f"Scenario: {scenario}")
    for result in results:
        print(result)
    print("\n")
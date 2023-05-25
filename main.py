import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from function import high_conditioned_elliptic, bent_cigar, discus, rosenbrocks, ackleys, weierstrass, griewanks, rastrigins, genetic_algorithm, differential_evolution


def calculate_max_NFC(num_dimension):
    return 3000 * num_dimension


# Run the experiments
benchmark_functions = [
    high_conditioned_elliptic,
    bent_cigar,
    discus,
    rosenbrocks,
    ackleys,
    weierstrass,
    griewanks,
    rastrigins
]

experiment_results = []
errors_dataframe = pd.DataFrame(columns=['Dimension', 'Algorithm', 'Benchmark',  'Mean', 'Best', 'Std'])

# Control Parameter Settings
population_num = 100
crossover_const = 0.9
scaling_factor = 0.8
mutation_prob = 0.01
dimension_values = [2, 10]
num_of_runs = 1

for benchmark_function in benchmark_functions:
    errors_diff_evo = []
    errors_gen_algo = []
    for num_dimension in dimension_values:
        errors_diff_evo_dim = []
        errors_gen_algo_dim = []
        for run_index in range(num_of_runs):
            diff_evo_error = differential_evolution(benchmark_function, num_dimension, population_num, crossover_const, scaling_factor, calculate_max_NFC(num_dimension))
            gen_algo_error = genetic_algorithm(benchmark_function, num_dimension, population_num, crossover_const, mutation_prob, calculate_max_NFC(num_dimension))
            errors_diff_evo_dim.append(diff_evo_error)
            errors_gen_algo_dim.append(gen_algo_error)

        errors_diff_evo.append(errors_diff_evo_dim)
        errors_gen_algo.append(errors_gen_algo_dim)

    # Performance Plots
    for dimension_index, num_dimension in enumerate(dimension_values):
        errors_diff_evo_dim = np.array(errors_diff_evo[dimension_index])
        errors_gen_algo_dim = np.array(errors_gen_algo[dimension_index])

        mean_error_diff_evo = np.mean(errors_diff_evo_dim, axis=0)
        mean_error_gen_algo = np.mean(errors_gen_algo_dim, axis=0)

        result_dict = {
            'Function': benchmark_function.__name__,
            'Dimension': num_dimension,
            'DE_Best_Error': np.min(errors_diff_evo_dim, axis=1),
            'DE_Mean_Error': mean_error_diff_evo,
            'GA_Best_Error': np.min(errors_gen_algo_dim, axis=1),
            'GA_Mean_Error': mean_error_gen_algo
        }

        experiment_results.append(result_dict)

        function_evaluation_counts = np.arange(len(mean_error_diff_evo)) * population_num

        plt.figure(figsize=(10, 6))
        sns.set_style('whitegrid')
        plt.plot(function_evaluation_counts, mean_error_diff_evo, label='DE', color='Orange', linewidth=2)
        plt.plot(function_evaluation_counts, mean_error_gen_algo, label='GA', color='Blue', linestyle='--', linewidth=2)
        plt.xlabel('NFCs', fontsize=12)
        plt.ylabel('Best Fitness Error So Far', fontsize=12)
        plt.title(f'{benchmark_function.__name__} for D={num_dimension}', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Fill the DataFrame with results
for result_dict in experiment_results:
    function_name = result_dict['Function']
    num_dimension = result_dict['Dimension']
    best_error_diff_evo = np.min(result_dict['DE_Best_Error'])
    mean_error_diff_evo = np.mean(result_dict['DE_Mean_Error'])
    std_error_diff_evo = np.std(result_dict['DE_Mean_Error'])
    best_error_gen_algo = np.min(result_dict['GA_Best_Error'])
    mean_error_gen_algo = np.mean(result_dict['GA_Mean_Error'])
    std_error_gen_algo = np.std(result_dict['GA_Mean_Error'])

    errors_dataframe = pd.concat([
        errors_dataframe,
        pd.DataFrame({
            'Dimension': num_dimension,
            'Algorithm': 'DE',
            'Benchmark': function_name,
            'Mean': mean_error_diff_evo,
            'Best': best_error_diff_evo,
            'Std': std_error_diff_evo
        }, index=[0]),
        pd.DataFrame({
            'Dimension': num_dimension,
            'Algorithm': 'GA',
            'Benchmark': function_name,
            'Mean': mean_error_gen_algo,
            'Best': best_error_gen_algo,
            'Std': std_error_gen_algo
        }, index=[0])
    ], ignore_index=True)

# Create DataFrames for each dimension
result_dataframe_list = []
for num_dimension in dimension_values:
    dimension_dataframe = errors_dataframe[errors_dataframe['Dimension'] == num_dimension]
    result_dataframe_list.append(dimension_dataframe)

# Display the DataFrames for each dimension
for dimension_index, num_dimension in enumerate(dimension_values):
    print(f"Error for Dimension {num_dimension}:")
    print(result_dataframe_list[dimension_index])

# Save the DataFrames to CSV files
for dimension_index, num_dimension in enumerate(dimension_values):
    csv_filename = f"result_dimension_{num_dimension}.csv"
    result_dataframe_list[dimension_index].to_csv(csv_filename, index=False)
    print(f"Saved DataFrame for Dimension {num_dimension} to {csv_filename}")
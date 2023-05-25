import numpy as np


# Benchmark Functions
def high_conditioned_elliptic(input_array):
    if isinstance(input_array, np.ndarray):
        return np.sum(100 * np.square(input_array))
    else:
        return 100 * np.square(input_array)


def bent_cigar(input_array):
    if isinstance(input_array, np.ndarray):
        return input_array[0] ** 2 + 1e6 * np.sum(np.square(input_array[1:]))
    else:
        return input_array ** 2


def discus(input_array):
    if isinstance(input_array, np.ndarray):
        return 1e6 * input_array[0] ** 2 + np.sum(np.square(input_array[1:]))
    else:
        return 1e6 * input_array ** 2


def rosenbrocks(input_array):
    if isinstance(input_array, np.ndarray):
        return np.sum([100 * (input_array[i + 1] - input_array[i] ** 2) ** 2 + (1 - input_array[i]) ** 2 for i in range(len(input_array) - 1)])
    else:
        return 0


def ackleys(input_array):
    if isinstance(input_array, np.ndarray):
        array_length = len(input_array)
        sum_1 = -0.2 * np.sqrt(np.sum(input_array ** 2) / array_length)
        sum_2 = np.sum(np.cos(2 * np.pi * input_array)) / array_length
        return -20 * np.exp(sum_1) - np.exp(sum_2) + 20 + np.exp(1)
    else:
        return -20 * np.exp(-0.2 * np.sqrt(input_array ** 2)) - np.exp(np.cos(2 * np.pi * input_array)) + 20 + np.exp(1)


def weierstrass(input_array):
    if isinstance(input_array, np.ndarray):
        a_coeff = 0.5
        b_coeff = 3
        k_max_val = 20
        outer_sum = 0
        for i in range(len(input_array)):
            inner_sum = 0
            for k in range(k_max_val + 1):
                inner_sum += a_coeff ** k * np.cos(2 * np.pi * b_coeff ** k * (input_array[i] + 0.5))
            outer_sum += inner_sum
        correction_sum = 0
        for k in range(k_max_val + 1):
            correction_sum += a_coeff ** k * np.cos(2 * np.pi * b_coeff ** k * 0.5)
        return outer_sum - len(input_array) * correction_sum
    else:
        a_coeff = 0.5
        b_coeff = 3
        k_max_val = 20
        outer_sum = 0
        inner_sum = 0
        for k in range(k_max_val + 1):
            inner_sum += a_coeff ** k * np.cos(2 * np.pi * b_coeff ** k * (input_array + 0.5))
        outer_sum += inner_sum
        correction_sum = 0
        for k in range(k_max_val + 1):
            correction_sum += a_coeff ** k * np.cos(2 * np.pi * b_coeff ** k * 0.5)
        return outer_sum - correction_sum


def griewanks(input_array):
    if isinstance(input_array, np.ndarray):
        return np.sum(input_array ** 2 / 4000) - np.prod(np.cos(input_array / np.sqrt(np.arange(1, len(input_array) + 1)))) + 1
    else:
        return input_array ** 2 / 4000 - np.cos(input_array / np.sqrt(np.arange(1, 2))) + 1


def rastrigins(input_array):
    if isinstance(input_array, np.ndarray):
        return 10 * len(input_array) + np.sum(input_array ** 2 - 10 * np.cos(2 * np.pi * input_array))
    else:
        return 10 * len([input_array]) + input_array ** 2 - 10 * np.cos(2 * np.pi * input_array)


def differential_evolution(target_function, dimensions, population_size=100, crossover_rate=0.9,
                           differential_weight=0.8, max_function_calls=3000):
    gene_pool = np.random.uniform(-10, 10, (population_size, dimensions))
    fitness_values = np.array([target_function(individual) for individual in gene_pool])
    best_fitness_history = []
    function_call_counter = 0

    while function_call_counter < max_function_calls:
        for individual_index in range(population_size):
            valid_indices = list(range(population_size))
            valid_indices.remove(individual_index)
            parent_1, parent_2, parent_3 = np.random.choice(valid_indices, size=3, replace=False)
            mutation_vector = gene_pool[parent_1] + differential_weight * (gene_pool[parent_2] - gene_pool[parent_3])
            crossover_condition = np.random.rand(dimensions) < crossover_rate
            child = np.where(crossover_condition, mutation_vector, gene_pool[individual_index])

            child_fitness_value = target_function(child)
            function_call_counter += 1
            if child_fitness_value < fitness_values[individual_index]:
                gene_pool[individual_index] = child
                fitness_values[individual_index] = child_fitness_value

            if function_call_counter >= max_function_calls:
                break

        best_fitness_history.append(np.min(fitness_values))

    return best_fitness_history


# Genetic Algorithm (GA)
def genetic_algorithm(objective_function, dimensions, population_size=100, crossover_rate=0.9,
                      mutation_probability=0.01, max_function_calls=3000):
    gene_pool = np.random.uniform(-10, 10, (population_size, dimensions))
    fitness_values = np.array([objective_function(individual) for individual in gene_pool])
    best_fitness_history = []
    function_call_counter = 0

    while function_call_counter < max_function_calls:
        children = np.empty_like(gene_pool)
        for individual_index in range(population_size):
            valid_indices = list(range(population_size))
            valid_indices.remove(individual_index)
            parent_1, parent_2, parent_3 = np.random.choice(valid_indices, size=3, replace=False)
            crossover_condition = np.random.rand(dimensions) < crossover_rate
            children[individual_index] = np.where(crossover_condition, gene_pool[parent_1] + mutation_probability * (
                        gene_pool[parent_2] - gene_pool[parent_3]), gene_pool[individual_index])

        children_fitness_values = np.array([objective_function(individual) for individual in children])
        function_call_counter += population_size

        for individual_index in range(population_size):
            if children_fitness_values[individual_index] < fitness_values[individual_index]:
                gene_pool[individual_index] = children[individual_index]
                fitness_values[individual_index] = children_fitness_values[individual_index]

            if function_call_counter >= max_function_calls:
                break

        best_fitness_history.append(np.min(fitness_values))

    return best_fitness_history

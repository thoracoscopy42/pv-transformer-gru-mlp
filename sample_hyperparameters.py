import random
from itertools import product


def sample_hyperparameters(search_space, number_of_trials, seed=42):
    parameter_names = list(search_space.keys())
    parameter_values = list(search_space.values())

    combinations = product(*parameter_values)

    configurations = [
        dict(zip(parameter_names, values))
        for values in combinations
    ]

    if not 1 <= number_of_trials <= len(configurations):
        raise ValueError(
            f"You need to choose a number between 1 and {len(configurations)}."
        )

    generator = random.Random(seed)

    return generator.sample(configurations, number_of_trials)
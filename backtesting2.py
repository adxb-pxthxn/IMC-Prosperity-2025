import itertools

def generate_param_combinations(param_grid):
    param_names = param_grid.keys()
    param_values = param_grid.values()
    combinations = list(itertools.product(*param_values))
    return [dict(zip(param_names, combination)) for combination in combinations]

param_grid = {
    "AMETHYSTS": {
        "fair_value": [10000],
        "take_width": [1],
        "clear_width": [0.5],
        "volume_limit": [0],
        # for making
        "disregard_edge": [1,2],  # disregards orders for joining or pennying within this value from fair
        "join_edge": [2],# joins orders within this edge 
        "default_edge": [4]
    },
    "STARFRUIT": {
        "take_width": [1],
        "clear_width": [0, -0.25],
        "prevent_adverse": [True],
        "adverse_volume": [15],
        "reversion_beta": [-0.229],
        # for making
        "disregard_edge": [1],
        "join_edge": [3],
        "default_edge": [5],
    },
}

print(generate_param_combinations(param_grid["STARFRUIT"]))
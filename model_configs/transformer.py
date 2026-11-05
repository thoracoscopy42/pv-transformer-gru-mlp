MODEL_SPACE = {
    "d_model": [32, 64, 128],
    "number_of_heads": [2, 4, 8],
    "number_of_layers": [1, 2, 3],
    "dim_feedforward": [64, 128, 256, 512],
    "dropout": [0.0, 0.1, 0.2, 0.3],
}

TRAINING_SPACE = {
    "learning_rate": [ 0.0001, 0.001, 0.00005, 0.0005, 0.0003, 0.003],
    "batch_size": [18, 32, 64, 128, 256, 512],
    "weight_decay": [0.0, 0.00001, 0.0001, 0.001],
}
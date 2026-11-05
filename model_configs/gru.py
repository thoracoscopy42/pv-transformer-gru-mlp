MODEL_SPACE = {
    "hidden_size": [32, 64, 128, 256],
    "number_of_layers": [1, 2, 3],
    "dense_size": [16, 32, 64, 128],
    "dropout": [0.0, 0.1, 0.2, 0.3],
}

TRAINING_SPACE = {
    "learning_rate": [ 0.0001, 0.001, 0.00005, 0.0005, 0.0003, 0.003],
    "batch_size": [18, 32, 64, 128, 256, 512],
    "weight_decay": [0.0, 0.00001, 0.0001, 0.001],
}
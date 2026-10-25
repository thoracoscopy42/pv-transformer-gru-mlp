MODEL_SPACE = {
    "hidden_sizes": [
        (64, 32),
        (128, 64),
        (256, 128),
        (128, 64, 32),
    ],
    "dropout": [0.0, 0.1, 0.2],
}

TRAINING_SPACE = {
    "learning_rate": [0.0001, 0.0005, 0.001],
    "batch_size": [64, 128, 256],
    "weight_decay": [0.0, 0.0001, 0.001],
}
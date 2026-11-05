from torch import nn


class MLPForecaster(nn.Module):
    def __init__(
        self,
        sequence_length,
        number_of_features,
        hidden_sizes=(128, 64),
        dropout=0.1,
    ):
        super().__init__()

        input_size = sequence_length * number_of_features

        layers = [
            nn.Flatten(start_dim=1)
        ]

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        number_of_features,
        hidden_size=64,
        number_of_layers=1,
        dense_size=32,
        dropout=0.1,
    ):
        super().__init__()

        gru_dropout = (
            dropout if number_of_layers > 1 else 0.0
        )

        self.gru = nn.GRU(
            input_size=number_of_features,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, X):
        sequence_output, _ = self.gru(X)

        last_output = sequence_output[:, -1, :]

        predictions = self.regression_head(last_output)

        return predictions
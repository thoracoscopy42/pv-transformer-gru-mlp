import torch
from torch import nn


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        sequence_length,
        number_of_features,
        d_model=64,
        number_of_heads=4,
        number_of_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % number_of_heads != 0:
            raise ValueError(
                "d_model musi być podzielne przez number_of_heads."
            )

        self.sequence_length = sequence_length

        self.input_projection = nn.Linear(
            number_of_features,
            d_model,
        )

        self.positional_embedding = nn.Parameter(
            torch.empty(1, sequence_length, d_model)
        )

        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02,
        )

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=number_of_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.normalization = nn.LayerNorm(d_model)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, X):
        if X.shape[1] != self.sequence_length:
            raise ValueError(
                "Niepoprawna długość okna wejściowego."
            )

        encoded = self.input_projection(X)
        encoded = encoded + self.positional_embedding

        for layer in self.encoder_layers:
            encoded = layer(encoded)

        encoded = self.normalization(encoded)

        last_output = encoded[:, -1, :]

        predictions = self.regression_head(last_output)

        return predictions
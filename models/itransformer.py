import torch
from torch import nn


class ITransformerForecaster(nn.Module):
    """Forecast one target using attention between variable histories."""

    def __init__(
        self,
        sequence_length,
        number_of_features,
        target_index,
        d_model=64,
        number_of_heads=4,
        number_of_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_window_normalization=True,
    ):
        super().__init__()

        if sequence_length < 1 or number_of_features < 1:
            raise ValueError(
                "Sequence length and number of features must be positive."
            )

        if d_model < 1 or number_of_heads < 1:
            raise ValueError(
                "Model dimension and number of heads must be positive."
            )

        if d_model % number_of_heads != 0:
            raise ValueError(
                "d_model must be divisible by number_of_heads."
            )

        if number_of_layers < 1 or dim_feedforward < 1:
            raise ValueError(
                "Number of layers and feedforward size must be positive."
            )

        if not 0 <= target_index < number_of_features:
            raise ValueError("Invalid target feature index.")

        self.sequence_length = sequence_length
        self.number_of_features = number_of_features
        self.target_index = target_index
        self.use_window_normalization = use_window_normalization

        # Each variable's entire history becomes one token.
        self.history_embedding = nn.Linear(
            sequence_length,
            d_model,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=number_of_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.normalization = nn.LayerNorm(d_model)

        # Predict one future value from the target variable's token.
        self.forecast_head = nn.Linear(d_model, 1)

    def forward(self, X):
        if X.ndim != 3:
            raise ValueError(
                "Expected input shape: (batch, sequence, features)."
            )

        if X.shape[1:] != (
            self.sequence_length,
            self.number_of_features,
        ):
            raise ValueError(
                "Input dimensions differ from model configuration."
            )

        if self.use_window_normalization:
            window_mean = X.mean(
                dim=1,
                keepdim=True,
            ).detach()

            centered = X - window_mean

            window_std = torch.sqrt(
                centered.var(
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            )

            X = centered / window_std

        # (batch, sequence, features) -> (batch, features, sequence)
        variable_histories = X.transpose(1, 2)

        # (batch, features, sequence) -> (batch, features, d_model)
        encoded = self.history_embedding(variable_histories)
        encoded = self.embedding_dropout(encoded)

        for layer in self.encoder_layers:
            encoded = layer(encoded)

        encoded = self.normalization(encoded)

        # Select the target token after it has attended to all variables.
        target_representation = encoded[:, self.target_index, :]

        predictions = self.forecast_head(target_representation)

        if self.use_window_normalization:
            target_mean = window_mean[:, 0, self.target_index].unsqueeze(1)
            target_std = window_std[:, 0, self.target_index].unsqueeze(1)

            predictions = predictions * target_std + target_mean

        return predictions
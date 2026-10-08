import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SelfAttention(nn.Module):
    def __init__(self, heads, embedding_dimension, input_bias, output_bias):
        super().__init__()
        if embedding_dimension % heads != 0:
            raise ValueError("embedding_dimension must be divisible by heads.")
        self.heads = heads
        self.head_dimension = embedding_dimension // heads
        self.input_projection = nn.Linear(
            embedding_dimension,
            3 * embedding_dimension,
            bias=input_bias,
        )
        self.output_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=output_bias,
        )

    def forward(self, tensor, causal_mask):
        batch_size, sequence_length, embedding_dimension = tensor.shape
        projected = self.input_projection(tensor)
        query, key, value = projected.chunk(3, dim=-1)
        shape = (batch_size, sequence_length, self.heads, self.head_dimension)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        weights = query @ key.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, -torch.inf)
        weights = functional.softmax(weights / math.sqrt(self.head_dimension), dim=-1)

        output = weights @ value
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            embedding_dimension,
        )
        return self.output_projection(output)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.normalization = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(
            heads=1,
            embedding_dimension=channels,
            input_bias=True,
            output_bias=True,
        )

    def forward(self, tensor):
        residual = tensor
        tensor = self.normalization(tensor)
        batch_size, channels, height, width = tensor.shape
        tensor = tensor.view(batch_size, channels, height * width).transpose(1, 2)
        tensor = self.attention(tensor, False)
        tensor = tensor.transpose(1, 2).view(batch_size, channels, height, width)
        return tensor + residual


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.normalization_one = nn.GroupNorm(32, input_channels)
        self.convolution_one = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.normalization_two = nn.GroupNorm(32, output_channels)
        self.convolution_two = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        if input_channels == output_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(input_channels, output_channels, 1, 1, 0)

    def forward(self, tensor):
        residual = self.residual_layer(tensor)
        tensor = self.convolution_one(functional.silu(self.normalization_one(tensor)))
        tensor = self.convolution_two(functional.silu(self.normalization_two(tensor)))
        return tensor + residual


class SpatialEncoder(nn.Module):
    def __init__(self, latent_channels, input_channels, log_variance_bounds):
        super().__init__()
        self.log_variance_bounds = log_variance_bounds
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(input_channels, 128, 3, 1, 1),
                ResidualBlock(128, 128),
                ResidualBlock(128, 128),
                nn.Conv2d(128, 128, 3, 2, 0),
                ResidualBlock(128, 256),
                ResidualBlock(256, 256),
                nn.Conv2d(256, 256, 3, 2, 0),
                ResidualBlock(256, 512),
                ResidualBlock(512, 512),
                nn.Conv2d(512, 512, 3, 2, 0),
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),
                ResidualBlock(512, 512),
                AttentionBlock(512),
                ResidualBlock(512, 512),
                nn.GroupNorm(32, 512),
                nn.SiLU(),
                nn.Conv2d(512, latent_channels, 3, 1, 1),
                nn.Conv2d(latent_channels, 2 * latent_channels, 1, 1, 0),
            ]
        )

    def forward(self, tensor):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                tensor = functional.pad(tensor, (0, 1, 0, 1))
            tensor = layer(tensor)
        mean, log_variance = torch.chunk(tensor, chunks=2, dim=1)
        log_variance = torch.clamp(
            log_variance,
            min=float(self.log_variance_bounds[0]),
            max=float(self.log_variance_bounds[1]),
        )
        return mean, log_variance


class SpatialDecoder(nn.Module):
    def __init__(self, latent_channels, scaling_factor, output_channels):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 1, 1, 0),
            nn.Conv2d(latent_channels, 512, 3, 1, 1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, output_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, latent):
        return self.layers(latent / self.scaling_factor)


class SpatialBetaVAE(nn.Module):
    def __init__(
        self,
        latent_channels,
        scaling_factor,
        log_variance_bounds,
        input_channels,
        output_channels,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.encoder = SpatialEncoder(
            latent_channels,
            input_channels,
            log_variance_bounds,
        )
        self.decoder = SpatialDecoder(
            latent_channels,
            scaling_factor,
            output_channels,
        )

    def reparameterize(self, mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(standard_deviation)
        return (mean + standard_deviation * noise) * self.scaling_factor

    def forward(self, tensor):
        mean, log_variance = self.encoder(tensor)
        latent = self.reparameterize(mean, log_variance)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, log_variance

    def reconstruct_from_mean(self, tensor):
        mean, log_variance = self.encoder(tensor)
        reconstruction = self.decoder(mean * self.scaling_factor)
        return reconstruction, mean, log_variance


class PlantSizeClassifier(nn.Module):
    def __init__(self, input_dimension, hidden_dimensions, class_count, dropout):
        super().__init__()
        layers = []
        previous_dimension = input_dimension
        for hidden_dimension in hidden_dimensions:
            layers.extend(
                [
                    nn.Linear(previous_dimension, int(hidden_dimension)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous_dimension = int(hidden_dimension)
        layers.append(nn.Linear(previous_dimension, class_count))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)

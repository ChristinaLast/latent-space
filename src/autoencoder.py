import torch.nn as nn
import torch
from src.encoder import Encoder
from src.decoder import Decoder, Up
from config.model_settings import EncoderConfig


class Autoencoder(nn.Module):
    def __init__(
        self, device,
    ) -> None:
        super().__init__()
        self.encoder = Encoder.from_dataclass_config(EncoderConfig).initialize_model()
        self.decoder = Decoder(device) # Pass the number of output channels of your encoder (2048 for ResNet50)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

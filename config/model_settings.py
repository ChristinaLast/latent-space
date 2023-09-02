import os
from dataclasses import field
from typing import Any, Dict, List, Sequence
import torch

from pydantic import StrictStr
from pydantic.dataclasses import dataclass


@dataclass
class LoadDataConfig:
    DATA_DIR = "/home/ariana/subjective_paths/data/"
    CITY_NAMES = ["seattle", "bridgeport"]
    BATCH_SIZE = 16


@dataclass
class EncoderConfig:
    MODEL_NAME = "resnet"
    FEATURE_EXTRACT = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainModelConfig:
    NUM_EPOCHS = 150
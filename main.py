import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.load_data import LoadData
from config.model_settings import LoadDataConfig, EncoderConfig, TrainModelConfig
from src.autoencoder import Autoencoder
from src.encoder import Encoder
from src.train import TrainModel
import gc
from math import sqrt 

gc.collect()
torch.cuda.empty_cache()

def main():
    model = "segment_anything",
    batch_size = 1
    # Initialize LoadData and execute
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = LoadData.from_dataclass_config(LoadDataConfig)
    dataloader = loader.execute()

    model_ft = Autoencoder(device).to(device)
    encoder = Encoder.from_dataclass_config(EncoderConfig)
    optimizer_ft = encoder.optimize_parameters(model_ft)
    scheduler = ReduceLROnPlateau(optimizer_ft, patience=10, factor=sqrt(0.1))

    train_model = TrainModel.from_dataclass_config(TrainModelConfig)
    train_model.execute(model_ft, dataloader, optimizer_ft, batch_size)

if __name__ == "__main__":
    main()
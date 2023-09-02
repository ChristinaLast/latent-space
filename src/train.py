import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError
import time
import os
import numpy as np
import copy
from functools import partial
from PIL import Image
from config.model_settings import TrainModelConfig
from torch.nn.parallel import DataParallel
from torch.utils.data import Subset

class TrainModel:
    def __init__(self, num_epochs: int, device: torch.device):
        self.num_epochs = num_epochs
        self.device = device

    @classmethod
    def from_dataclass_config(cls, config: TrainModelConfig) -> "TrainModel":
        return cls(
            num_epochs=config.NUM_EPOCHS,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
        )

    def execute(self, model, dataloader, optimizer, scheduler, accumulation_steps=2):
        criterion = nn.MSELoss()

        # Detect if we have a GPU available
        model_ft, hist, val_data = self.train_model(
            model,
            dataloader, 
            criterion,
            optimizer,
            scheduler,
        )
        model_ft = DataParallel(model_ft)  # Data Parallelism
        # Setup the loss fxn
        torch.save(model_ft.state_dict(), f"/home/last/latentspace/latent-space/models/{model_name}_{self.num_epochs}")

        return self.num_epochs, model_ft, hist, val_data

    # Train and evaluate
    def train_model(
        self,
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        is_inception=False,
    ):
        since = time.time()

        val_loss_history = []
        train_loss_history = []

        best_loss = 1.0
        model = model.to(self.device)

        for epoch in range(self.num_epochs):
            print("Epoch {}/{}".format(epoch, self.num_epochs - 1))
            print("-" * 10)
            gradient_dict = {}
            # Each epoch has a training and validation phase
            
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                for i, batch in enumerate(dataloader[phase]):
                    inputs = batch[0].to(self.device)
                    # image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
                    with torch.no_grad():
                        labels = batch[0].to(self.device)

                    # inputs, labels = inputs.cuda(), labels.cuda()
                    optimizer.zero_grad()
                    print("Model device:", next(model.parameters()).device)
                    print("Inputs device:", inputs.device)
                    print("Labels device:", labels.device)
                    # forward
                    # track history if only in train
                    print("mem 1", torch.cuda.memory_summary())
                    with torch.set_grad_enabled(phase == "train"):
                        if is_inception and phase == "train":
                            outputs, aux_outputs = model(
                                inputs
                            )
                            
                        else:
                            print("mem 2", torch.cuda.memory_summary())
                            outputs = checkpoint(model, inputs)
                        
                        loss = criterion(outputs, labels.to(self.device))
                        
                        _, preds = torch.max(outputs, 1)
                        del inputs
                        del labels
                        torch.cuda.empty_cache()
                        
                        if phase == 'train':
                            loss = loss / accumulation_steps  # Normalize loss
                            loss.backward()
                            if (i+1) % accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                    running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / len(
                    dataloader[phase].dataset
                )
                
                if phase == "train":
                    train_loss_history.append(epoch_loss)
                    scheduler.step(epoch_loss)
                    print("Outputs shape before permute:", outputs.shape)
                    optimizer.step()

                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if phase == "val":
                    val_loss_history.append(epoch_loss)
                    print("Learning rate: {:.4e}".format(scheduler._last_lr[0]))
                    plt.figure()
                    plt.plot(np.array(train_loss_history))
                    plt.show()
                    plt.savefig(f"/loss_{i}.png")
                    plt.close()
                    
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Loss: {:4f}".format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), "/home/last/latentspace/models/best_model.pth")
        return model, val_loss_history, dataloader[phase]

import os
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim

from config.model_settings import EncoderConfig
from transformers import SamModel, SamProcessor

class Encoder:
    def __init__(
        self,
        model_name: str,
        feature_extract: bool,
        device: str,
    ):
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.device = device

    @classmethod
    def from_dataclass_config(cls, config: EncoderConfig) -> "Encoder":
        return cls(
            model_name=config.MODEL_NAME,
            feature_extract=config.FEATURE_EXTRACT,
            device=config.DEVICE,
        )

    def initialize_model(self, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if self.model_name == "resnet":
            """resnet"""
            self.num_classes = 750
            model_ft = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
            model_ft = self.remove_final_layer(model_ft)
        
        elif self.model_name == "segment_anything":
            """Segment-Anything model"""
            model_ft = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
            
            # model_ft = nn.Sequential(*list(model.children())[:2])
            # self.set_parameter_requires_grad(model_ft, self.feature_extract)
            # num_ftrs = model_ft.in_features  # replace 'some_final_layer' with actual layer name
            # model_ft.some_final_layer = nn.Linear(num_ftrs, self.num_classes)
            # input_size = 256  # replace this with the actual input size
            # model_ft = self.remove_final_layer(model_ft)

        elif self.model_name == "alexnet":
            """Alexnet"""
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 256
            model_ft = self.remove_final_layer(model_ft)


        elif self.model_name == "vgg":
            """VGG11_bn"""
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
            model_ft = self.remove_final_layer(model_ft)

        elif self.model_name == "squeezenet":
            """Squeezenet"""
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(
                512, self.num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            model_ft.num_classes = self.num_classes
            input_size = 224
            model_ft = self.remove_final_layer(model_ft)

        elif self.model_name == "densenet":
            """Densenet"""
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
            model_ft = self.remove_final_layer(model_ft)
        
        elif self.model_name == "inception":
            """Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 299
    
        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft

    def set_parameter_requires_grad(self, model, feature_extracting):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def optimize_parameters(self, model_ft):
        # Send the model to GPU
        model_ft = model_ft.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        print("Params to learn:")
        params_to_update = []
        if self.feature_extract:
            if self.model_name == "segment_anything":
                # make sure we only compute gradients for mask decoder
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if "mask_decoder" in name or "prompt_encoder" in name or "decoder" in name:
                        param.requires_grad_(False)
                
                for name, param in model_ft.named_parameters():
                    if name.startswith("encoder.vision_encoder.neck.layer_norm2.bias"):
                        print(name)
                        print("\t", name)
                        param.requires_grad_(True)
                        params_to_update.append(param)
            else:
                for param in model_ft.parameters():
                    param.requires_grad = False
                    params_to_update.append(param)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == False:
                    print("\t", name)

        # Observe that all parameters are being optimized
        # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(params_to_update, lr=0.001)
        return optimizer_ft

    def remove_final_layer(self, model_ft):
        """
        Remove the final layer of the model

        Parameters
        ----------
        model_ft : torch.nn.Module
            Model to remove the final layer
        Returns
        -------
        new_model_ft : torch.nn.Module
            Model without the final layer
        """
        new_model_ft = torch.nn.Sequential(*(list(model_ft.children())[:-1]))

        return new_model_ft

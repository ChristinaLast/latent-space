from functools import partial
from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import SamProcessor

class ImageDataset(Dataset):
    def __init__(self, image_folder_list):
        self.image_folder_list = image_folder_list
        self.preprocess = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def __len__(self):
        return len(self.image_folder_list)

    def __getitem__(self, index):
        folder_path = self.image_folder_list[index]
        images = []
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.jpg'):
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path).convert('RGB')
                if self.model_name =="segment_anything":
                    img = self.preprocess(img)
                    tensor_img = torch.tensor(img['pixel_values'])
                    images.append(tensor_img)

                else:
                    preprocess = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.4786, 0.4747, 0.4467],
                                std=[0.0308, 0.0252, 0.0288],
                            ),
                        ]
                    )
                    images.append(preprocess(img))

        image_stack = torch.cat(images, dim=0)
        return image_stack


class LoadData:
    def __init__(self, data_path: str, city_names: list, batch_size: int):
        self.data_path = data_path
        self.city_names = city_names
        self.batch_size = batch_size

    @classmethod
    def from_dataclass_config(cls, config) -> "LoadData":
        return cls(data_path=config.DATA_DIR, city_names=config.CITY_NAMES, batch_size=config.BATCH_SIZE)

    def list_all_subdirectories(self, base_path):
        subdirectories = []
        for root, dirs, files in os.walk(base_path):
            if not dirs:  # This means it's a leaf directory
                subdirectories.append(root)
        return subdirectories

    def execute(self):
        all_subdirectories = []
        for city in self.city_names:
            city_base_dir = os.path.join(self.data_path, f"img_{city}")
            city_subdirs = self.list_all_subdirectories(city_base_dir)
            all_subdirectories.extend(city_subdirs)

        dataset = ImageDataset(all_subdirectories)
        
        # Split into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        dataloaders = {
            'train': DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=5
            ),
            'val': DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=0
            )
        }

        return dataloaders

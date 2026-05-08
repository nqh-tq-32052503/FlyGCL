import os 

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class TorchvisionDataset:
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    roots = {
        "tinyimagenet" : "TinyImage",
        "cub200" : "CUB200",
        "imagenet-r" : "imagenet-r",
        "eurosat" : "eurosat",
        "cropdisease" : "CropDisease",
        "mnist" : "MNIST",
        "chestx" : "ChestX",
        "resisc45" : "resisc45",
    }
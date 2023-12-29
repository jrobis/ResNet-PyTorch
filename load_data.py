import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# from config import config
# from config.config import model_config

from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yml')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.training.train_batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=True, transform=transform)

test_val_split = 0.8

testset, valset = random_split(testset,[8000,2000])

testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.training.eval_batch_size,
                                         shuffle=False, num_workers=2)

valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.training.eval_batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
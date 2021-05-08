import torch 
import torch.nn as nn 
from torch.autograd import Variable

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.encoder=torch.nn.Sequential(
            nn.Linear(784,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,200),
            nn.ReLU())
        self.fc1=nn.Linear(200,100)
        self.fc2=nn.Linear(200,10)
        self.decoder=nn.Sequential(
            nn.Linear(10,200),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,784),
            nn.Sigmoid())
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
        self.fc1=nn.Linear(200,10)
        self.fc2=nn.Linear(200,10)
        self.decoder=nn.Sequential(
            nn.Linear(10,200),
            nn.ReLU(),
            nn.Linear(200,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,784),
            nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
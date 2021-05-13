import yaml 
import torch  
import torch.nn.functional as F

from model import get_model
from dataset import get_dataset
from optimizer import get_optimizer
from torch.utils.data import DataLoader
class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg 
        self.model=get_model(cfg['model']['name'])()
        if self.cfg['model']['cuda']:
            self.model=self.model.cuda()
        self.optimizer = get_optimizer(cfg['training']['optimizer']['name'])(self.model.parameters(),lr=1e-3)
        train_dataset= get_dataset(cfg['dataset']['name'])(cfg)
        self.data_loader=DataLoader(train_dataset,batch_size=cfg['dataset']['batch_size'],shuffle=True,num_workers=2)
    def train_autoencoder(self):
        for epoch in range(self.cfg['training']['epoches']):
            for batch_idx,(imgs,labels) in enumerate(self.data_loader):
                #imgs=imgs.view(imgs.shape[0],-1)
                recon_batch, mu, logvar = self.model(imgs)
                loss=self.loss_func(recon_batch,imgs,mu,logvar)
                loss.backward()
                self.optimizer.zero_grad()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(imgs), len(self.data_loader.dataset),
                        100. * batch_idx / len(self.data_loader),
                        loss.item() / imgs.shape[0]))
    def train_dec(self):
        pass 

    def loss_func(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= 784 * 128
        return BCE + KLD

    def run(self):
        self.train_autoencoder()
        self.train_dec()
        pass 

def main():
    with open('configs/mnist.yml')  as fp:
        cfg=yaml.load(fp,Loader=yaml.FullLoader)
        trainer=Trainer(cfg)
        trainer.run()

if __name__=='__main__':
    main()
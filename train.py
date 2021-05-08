from model import get_model
from optimizer import get_optimizer
class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg 
        self.model=get_model(cfg['model']['name'])()
        if self.cfg['model']['cuda']:
            self.model=self.model.cuda()
        self.optimizer = get_optimizer(cfg['training']['optimizer']['name'])(self.model.parameters(),lr=1e-3)
        train_dataset=
        self.data_loader=DataLoader(train_dataset)
    def train_autoencoder(self):
        for i in range(self.cfg['training']['epoches']):
            for batch,(imgs,labels) in enumerate(self.data_loader):
                recon_batch, mu, logvar = self.model(imgs)
                
        pass 
    
    def train_dec(self):
        pass 

    def run(self):
        self.train_autoencoder()
        self.train_dec()
        pass 
import yaml 

from torch import nn
from model import get_model
from dataset import get_dataset
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg 
        self.model=get_model(cfg['model']['name'])(cfg)
        if self.cfg['model']['cuda']:
            self.model=self.model.cuda()
        train_dataset= get_dataset(cfg['dataset']['name'])(cfg)
        self.data_loader=DataLoader(train_dataset,batch_size=cfg['dataset']['batch_size'],shuffle=True,num_workers=2)
        self.loss=nn.MSELoss(reduction="mean")
        self.threshold = cfg["upper_threshold"]
    def run(self):
        
        epoches=self.cfg['training']['epoches']
        for epoch in range(epoches):
            for batch_idx,(imgs,labels) in enumerate(self.data_loader): 
                
                print('\r Epoch:{}/{} '.format(epoch,epoches) ,end='',flush=True)

if __name__=='__main__':
    with open('configs/mnist_dac.yml')  as fp:
        cfg=yaml.load(fp,Loader=yaml.FullLoader)
    trainer=Trainer(cfg)
    trainer.run()
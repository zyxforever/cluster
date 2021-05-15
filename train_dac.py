import yaml 
import numpy as np 
import torch 
import torch.nn.functional as F

#from tools.utils import ACC

from tqdm import tqdm
from torch import nn
from model import get_model
from dataset import get_dataset
from optimizer import get_optimizer
from utils.metrics import Scores
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg 
        self.model=get_model(cfg['model']['name'])()
        if self.cfg['model']['cuda']:
            self.model=self.model.cuda()
        self.optimizer=get_optimizer(cfg['training']['optimizer']['name'])(self.model.parameters(),lr=cfg['training']['optimizer']['lr'])
        train_dataset= get_dataset(cfg['dataset']['name'])(cfg)
        self.data_loader=DataLoader(train_dataset,batch_size=cfg['dataset']['batch_size'],shuffle=True,num_workers=2)
        self.loss=nn.MSELoss(reduction="mean")
        self.threshold = cfg['training']["upper_threshold"]
        self.val_scores = Scores(10, 10)
    def eval(self):
        self.model.eval()
        pre_y=[]
        tru_y=[]
        for x,y in self.data_loader:
            x=x.view(-1,1,28,28)
            if self.cfg['model']['cuda']:
                x=x.cuda()
            f=self.model(x)
            pre_y.append(torch.argmax(f,1).detach().cpu().numpy())
            tru_y.append(y.numpy())
        pre_y=np.concatenate(pre_y,0)
        tru_y=np.concatenate(tru_y,0)
        self.val_scores.update(tru_y,pre_y)
        self.val_scores.compute()
        names = list(filter(lambda name: 'cls' not in name, self.val_scores.names))
        for name in names:
            print("name:{},val:{}".format(name,self.val_scores[name]))
    def run(self):
        Lambda=0
        epoches=self.cfg['training']['epoches']
        self.bar=tqdm(range(epoches))
        for epoch in self.bar:
            u=0.95-Lambda
            l=0.455+0.1*Lambda
            self.model.train()
            for batch_idx,(imgs,labels) in enumerate(self.data_loader): 
                x=imgs.view(-1,1,28,28)
                if self.cfg['model']['cuda']:
                    x=x.cuda()
                f=self.model(x)
                f_norm=F.normalize(f,p=2,dim=1)
                I=f_norm.mm(f_norm.t())
                loss=-torch.mean((I.detach()>u).float()*torch.log(torch.clamp(I,1e-10,1))+(I.detach()<l).float()*torch.log(torch.clamp(1-I,1e-10,1)))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.bar.set_description("Epoch:{}/{} Batch:{}/{}".format(epoch,epoches,batch_idx,len(self.data_loader)))
                #print('\r Epoch:{}/{} Batch:{}/{}'.format(epoch,epoches,batch_idx,len(self.data_loader)) ,end='',flush=True)
            Lambda+=1.1*0.009
            print('Acc:{}'.format(self.eval()),end='',flush=True)
if __name__=='__main__':
    with open('configs/mnist_dac.yml')  as fp:
        cfg=yaml.load(fp,Loader=yaml.FullLoader)
    trainer=Trainer(cfg)
    trainer.run()
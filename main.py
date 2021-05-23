import yaml
from dataset import get_dataset
def main():
    with open('configs/mnist_dac.yml') as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader) 
    
    dataset=get_dataset(cfg['dataset']['name'])()
    
    for e in range(cfg['training']['epoches']):
        print(e)
if __name__=='__main__':
    main()
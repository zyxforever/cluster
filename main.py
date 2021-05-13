

from train_dec import Trainer
def main():
    with open('configs/mnist.yml')  as fp:
        cfg=yaml.load(fp,Loader=yaml.FullLoader)
        trainer=Trainer(cfg)
        trainer.run()

if __name__=='__main__':
    main()
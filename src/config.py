import argparse

class Config:
    def __init__(self):
        """
        parser: to read all config
        config: save config in pairs like key:value
        """
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(description='uncertainty')
        self.config = dict()
        self.args = None
        self._add_default_setting()
        self.args = self.parser.parse_args()
        self._load_default_setting()
    def _add_default_setting(self):
        self.parser.add_argument('--cuda', default='0',help="cuda visible device")
        self.parser.add_argument("--learning_rate", default=1e-3, type=float,help="learning rate")
        self.parser.add_argument("--n_epochs", default=20, type=int,
                                 help="n epochs to train")
        self.parser.add_argument("--data_dir", default='/home/zyx/datasets', type=str,
                                 help="n epochs to train")
        self.parser.add_argument("--dataset",default="MNIST",type=str)
    def _load_default_setting(self):
        self.config['cuda'] = self.args.cuda
        self.config["learning_rate"] = self.args.learning_rate
        self.config['n_epochs'] = self.args.n_epochs
        self.config['dataset']=self.args.dataset
        self.config['data_dir']=self.args.data_dir
    def print_config(self):
        print('>' * 10, 'basic setting start', '>' * 20)
        for arg in self.config:
            print('{:20}: {}'.format(arg, self.config[arg]))
        print('>' * 10, 'basic setting end', '>' * 20)

    def get_config(self):
        """return config"""
        self.print_config()
        return self.config

if __name__=='__main__':
    cfg=Config().get_config()
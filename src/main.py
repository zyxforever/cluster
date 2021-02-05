import os 
import faiss 
import numpy as np
import scipy.io as sio

from config import Config
def main():
    cfg=Config().get_config()
    vec=sio.loadmat('/home/zyx/datasets/MNIST10k.mat')
    labels=vec['labels']
    data=vec['fea']
    d = data.shape[1]
    nmb_clusters=10
    clus = faiss.Clustering(d, nmb_clusters)

    clus.seed = np.random.randint(1234)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    res = faiss.StandardCpuResources()
    flat_config = faiss.IndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

if __name__=='__main__':
    main()
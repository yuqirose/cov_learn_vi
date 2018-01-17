""" Data loader func
Jan, 2018 Rose Yu @Caltech 
"""
import numpy as np
import numpy.random as npr
import torch
from torch.utils.data import Dataset

class SynthDataset(Dataset):
    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform

        #periodic
        self.ts, self.data=np.load('data/syn_periodproc.npy')
        self.num_exp,self.N,self.D=self.data.shape     
        self.data=self.data.reshape((self.num_exp,-1)) # (num_exp, ND)

        print('periodic data loaded...', 'n_exp',  self.num_exp, 'N', self.N, 'D',self.D)
        # Generate sample statistics
        # self.phi = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
        # self.nu = 5
        # self.covs = np.array([ inv_wishart_rand(self.nu,self.phi) for i in range(self.num_cov)])


    def __len__(self):
        return self.num_exp

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Generate samples on the fly
        # cov = self.covs[cov_idx]
        # y_i = np.array( [ npr.multivariate_normal(np.zeros((3,)), cov) for j in range(self.num_exp) ] )
        # sample = y_i[idx%self.num_exp,]

        if self.transform:
            sample = self.transform(sample)
        return torch.FloatTensor(sample), torch.FloatTensor(sample)



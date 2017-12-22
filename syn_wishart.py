import numpy as np
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from torch.utils.data import Dataset
import torch


class SynthDataset(Dataset):
    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform
        self.num_cov = int(1e2)
        self.num_exp = int(1e3)

        self.data=np.load('data/syn_wishart.npy')

        # Generate sample statistics
        # self.phi = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
        # self.nu = 5
        # self.covs = np.array([ inv_wishart_rand(self.nu,self.phi) for i in range(self.num_cov)])


    def __len__(self):
        return self.num_cov * self.num_exp

    def __getitem__(self, idx):

        cov_idx = int(idx/self.num_exp)
        data_i = self.data[cov_idx]
        sample = data_i[idx%self.num_exp,]
        # Generate samples on the fly
        # cov = self.covs[cov_idx]
        # y_i = np.array( [ npr.multivariate_normal(np.zeros((3,)), cov) for j in range(self.num_exp) ] )
        # sample = y_i[idx%self.num_exp,]

        if self.transform:
            sample = self.transform(sample)
        return torch.FloatTensor(sample), torch.FloatTensor(sample)





def inv_wishart_rand_prec(nu,phi):
    return inv(wishart_rand(nu,phi))

def inv_wishart_rand(nu, phi):
    # X ~ inv_wishart iff X^-1 ~ wishart
    return inv(wishart_rand(nu, inv(phi)))

def wishart_rand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = npr.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


    
if __name__ == '__main__':
    npr.seed(1)
    nu = 5
    num_cov = int(1e2)
    num_exp = int(1e3)
    phi = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
    print(phi)

    covs = np.array([ inv_wishart_rand(nu,phi) for i in range(num_cov)])
    print(covs.shape)

    print(np.mean(covs,0),"\n")

    y = []
    # generate obvs
    for i in range(num_cov):
        cov = covs[i,]
        y_i = np.array( [ npr.multivariate_normal(np.zeros((3,)), cov) for j in range(num_exp) ] )
        y.append(y_i)

    print(len(y))
    print(y[0].shape)

    np.save('data/syn_wishart.npy', y)


   

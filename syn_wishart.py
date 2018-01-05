import numpy as np
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt 
import seaborn as sns




class SynthDataset(Dataset):
    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform
        self.num_exp = int(1e4)

        self.data=np.load('data/syn_mixture.npy')


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

def gen_inv_wishart():
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

def gen_mvn():
    npr.seed(1)
    cov = np.array([[1,0],[0,1]])
    num_exp = int(1e4)

    y =  np.array( [ npr.multivariate_normal(np.ones((2,)), cov) for j in range(num_exp) ] )
    print(y.shape)
    np.save('data/syn_mvn.npy', y)

def gen_mixture():
    """ generate mixture of gaussian """
    npr.seed(1)
    num_exp = int(1e4)
    x_dim = 2
    z_dim = 2
    mu1 = [5, 5,]
    mu2 = [-5, -5]
    theta = np.array([[2,1],[-1,-2]])
    sigma = 0.1
    u = npr.uniform((num_exp,))
    z = np.zeros((num_exp, z_dim))
    cov = np.zeros((z_dim, z_dim))
    np.fill_diagonal(cov, 1)
    sz = int(num_exp/2)
    z[:sz, ]= npr.multivariate_normal(mu1, cov,sz)
    z[sz:, ] = npr.multivariate_normal(mu2,cov,sz)
    mu_x = theta@z.transpose()

    x = np.zeros((num_exp, x_dim))
    for i in range(num_exp):
        x[i] = npr.multivariate_normal(mu_x[:,i], sigma*cov)
    print(x.shape)
    np.save('data/syn_mixture.npy', x)

if __name__ == '__main__':
    # gen_mvn()
    # gen_mixture()
    y  = np.load('data/syn_mixture.npy')

    # plt.scatter(y[:,0], y[:,1])
    # plt.show()
    sns.kdeplot(y[:,0], y[:,1], color="b", shade=True)
    plt.show()

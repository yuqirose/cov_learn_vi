from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np



class DGP(nn.Module):
    """deep gaussian process """
    def __init__(self, x_dim, h_dim, z_dim):
        super(DGP, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        # feature
        self.fc1 = nn.Linear(x_dim, h_dim)
        # encode
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)
        self.fc13 = nn.Linear(h_dim, z_dim)
         
        # encode 2
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)


        # transform
        self.fc3 = nn.Linear(z_dim, h_dim)
        # decode
        self.fc31 = nn.Linear(h_dim, x_dim)
        self.fc32 = nn.Linear(h_dim, x_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode_1(self, x):
        h1 = self.relu(self.fc1(x))
        enc_mean = self.fc11(h1)
        enc_cov_1 = self.fc12(h1)
        enc_cov_2 = self.fc13(h1)
        return enc_mean, enc_cov_1, enc_cov_2

    def encode_2(self, x):
        # adding GP prior on xi
        h2 = self.relu(self.fc2(x))
        enc_mean = self.fc21(h2)
        enc_cov = self.fc21(h2)
        return enc_mean, enc_cov


    def reparameterize_nm(self, mu, logcov):
        #  mu + R*esp (C = R'R)
        if self.training:
            cov = logcov.exp_() 
            eps = Variable(cov.data.new(cov.size()[0:2]).normal_())
            z = cov.mul(eps).add_(mu)
            return z
        else:
            return mu

    def reparameterize_gp(self, mu, logcov, u):
        #  sample from gaussian process 
        #  f = K^{-1} + L\eps \eps ~ N(0,I)
        if self.training:
            cov = torch.diag(logcov.exp_())+ torch.bmm(u.unsqueeze(2), u.unsqueeze(1)) # rank-1 approximation
            A_LU, pivots = torch.btrifact(cov.data)
            P, a_L, a_U = torch.btriunpack(A_LU, pivots)
            eps = Variable(cov.data.new(cov.size()[0:2]).normal_())
            eps_view = eps.unsqueeze(2)
            L = Variable(a_L)
            z = L.bmm(eps_view).squeeze(2).add(mu)
            return z
        else:
            return mu


    def decode(self, z):
        # p(x|z)~ N(f(z), \sigma )
        h3 = self.relu(self.fc3(z))
        dec_mean = self.fc31(h3)
        dec_cov = self.fc32(h3)
        return dec_mean, dec_cov

    def forward(self, x):
        # r(f|x)
        f_mean, f_cov_1, f_cov_2  = self.encode_1(x.view(-1, self.x_dim))
        f = self.reparameterize_gp(f_mean, f_cov_1, f_cov_2)

        # q(z|f)
        z_mean, z_cov = self.encode_2(f)
        z = self.reparameterize_nm(z_mean, z_cov)
        # p(x|z)
        x_mean, x_cov = self.decode(z)

        kld_loss = self._kld_loss(z_mean, z_cov)
        nll_loss = self._nll_loss(x_mean, x_cov, x)

        return kld_loss, nll_loss,(z_mean, z_cov), (x_mean, x_cov)

    def sample_z(self, x):
        # encoder 
        f_mean, f_cov_1, f_cov_2 = self.encode_1(x)
        f = self.reparameterize_gp(f_mean, f_cov_1, f_cov_2)

        z_mean, z_cov = self.encode_2(f)
        z = self.reparameterize_nm(z_mean, z_cov)
        return z.data.numpy()

    def sample_x(self):
        means = []
        covs = []
        
        z = Variable(torch.zeros(100, self.z_dim).normal_())        
        dec_mean, dec_cov = self.decode(z)
        # print(dec_mean)
        avg_mean = torch.mean(dec_mean, dim=0)
        avg_cov  = torch.mean(dec_cov, dim=0).exp()
        return avg_mean, avg_cov

    def _kld_loss(self, mu, logcov):
        # q(z|x)||p(z), q~N(mu1,S1), p~N(mu2,S2), mu1=mu, S1=cov, mu2=0, S2=I
        # 0.5 * (log 1 - log prod(cov) -d + sum(cov) + mu^2)
        KLD = 0.5 * torch.sum( -logcov -1 + logcov.exp()+ mu.pow(2))
        # Normalise by same number of elements as in reconstruction
        batch_size = mu.size()[0]
        KLD /= batch_size
        return KLD

    # def _kld_loss(self, mu, logcov):
    #     # 0.5 * log det(S2) - log det(S1) -d + trace(S2^-1 S1) + (mu2-mu1)^TS2^-1(mu2-mu1)
    #     cov = logcov.exp()
    #     prior_cov = prior_cov.exp()
    #     cov_inv = batch_inverse(cov)

    #     KLD = batch_trace(logcov) - batch_trace(prior_cov)
    #     KLD = KLD + batch_trace(torch.bmm(cov_inv,prior_cov))
    #     KLD = KLD + torch.bmm(torch.bmm(mu.unsqueeze(1), cov_inv), mu.unsqueeze(2))
    #     print('KLD', KLD)
    #     return -0.5*torch.sum(KLD)

    def _nll_loss(self, mean, cov, x): 
        # 0.5 * log det (x) + mu s
        # take one sample, reconstruction loss
        # print('mean', mean)
        # print('x', x)
        criterion = nn.MSELoss()

        NLL = criterion(mean, x)
        # NLL= 0.5 * torch.sum( mean.size()[1]*logcov + 1.0/logcov.exp() * (x-mean).pow(2))
        batch_size = mean.size()[0]
        NLL /= batch_size
        return NLL


    def _bce_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim))
        return BCE


def batch_trace(X):
    val = Variable(torch.zeros(X.size()[0],1))
    for i in range(X.size()[0]):
        trace_val = np.trace(X[i,:,:].data.numpy()).astype(float)
        val[i] = trace_val
    return val

def batch_inverse(X):
    val = Variable(torch.zeros(X.size()))
    for i in range(X.size()[0]):
        val[i,:,:] =X[i,:,:].inverse()
    return val








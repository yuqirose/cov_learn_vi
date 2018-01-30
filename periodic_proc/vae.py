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
import sys
sys.path.append("../")
from util.matutil import *
from util.batchutil import *

"""
Modified by
Shiwei Lan @ CalTech, 2018
version 1.3
"""

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim # ND
        self.h_dim = h_dim
        self.z_dim = z_dim # ND^*
        self.t_dim = np.int(x_dim/(2*z_dim/x_dim-1)) # N
        # feature
        self.fc0 = nn.Linear(x_dim, h_dim)
        # encode
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, np.int(self.t_dim*(self.t_dim+1)/2))
#         self.fc23 = nn.Linear(h_dim, z_dim)
        # transform
        self.fc3 = nn.Linear(z_dim, h_dim)
        # decode
        self.fc41 = nn.Linear(h_dim, x_dim)
        self.fc42 = nn.Linear(h_dim, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # problem-specific parameters
        self.D = np.int(self.z_dim/self.x_dim*2-1)
        self.N = np.int(self.x_dim/self.D)
        # GP kernel
        t = torch.linspace(0,2,steps=self.N+1); t = t[1:]
        self.K = Variable(torch.exp(-torch.pow(t.unsqueeze(1)-t.unsqueeze(0),2)/2/2) + 1e-4*torch.eye(self.N))
        self.Kh = torch.potrf(self.K)
#         self.iK = Variable(torch.inverse(self.K.data))
        self.iK = torch.potri(self.Kh)

    def encode(self, x):
        """ p(z|x)
            C = D + uu'
        """
        h1 = self.relu(self.fc0(x))
        enc_mean = self.fc21(h1)
        enc_covh = self.fc22(h1)
        #enc_cov_2 = self.fc23(h1)
        return enc_mean, enc_covh

    def reparameterize(self, mu, covh):
        #  mu + R*esp (C = R'R)
        if self.training:
            b_sz = mu.size()[0]
            eps = Variable(mu.data.new(mu.size()).normal_()).view(b_sz,self.t_dim,-1)
            covh_sqform = bivech(covh)
            z = covh_sqform.bmm(eps).view(b_sz,-1).add(mu)
            return z
        else:
            return mu

    def decode(self, z):
        # p(x|z)~ N(f(z), \sigma )
        h3 = self.relu(self.fc3(z))
        dec_mean = self.fc41(h3)
        dec_cov = self.fc42(h3)
        return dec_mean, dec_cov

    def forward(self, x):
        # encoder 
        enc_mean, enc_covh = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_covh)

        # decoder
        dec_mean, dec_cov = self.decode(z)

        kld_loss = self._kld_loss(enc_mean, enc_covh)
        nll_loss = self._nll_loss(dec_mean, dec_cov, x)

        return kld_loss, nll_loss,(enc_mean, enc_covh), (dec_mean, dec_cov)

    def sample_z(self, x):
        # encoder 
        enc_mean, enc_covh = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_covh)
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

    def _kld_loss(self, mu, covh):
        # q(z|x)||p(z), q~N(mu0,S0), p~N(mu1,S1), mu0=mu, S0=cov, mu1=0, S1=I
        # KLD = 0.5 * ( log det(S1) - log det(S0) -D + trace(S1^-1 S0) + (mu1-mu0)^TS1^-1(mu1-mu0) )
        
        cov = bivech2(covh)
        D_star = np.int(self.D*(self.D+1)/2)
        tr = D_star*torch.sum(torch.mul(self.iK,cov))
        vechI = Variable(th_vech(torch.eye(self.D))).view(1,1,-1)
        b_sz = mu.size()[0]
        I_mu = vechI - mu.view(b_sz,self.N,-1)
        quad = torch.sum( torch.mul(torch.matmul(self.iK,I_mu), I_mu) )
        ldet1 = 2*b_sz*D_star*torch.sum(torch.log(self.Kh.diag().abs()))
#         diag_idx = th_ivech(torch.arange(covh.data.size()[-1])).diag().long()
        diag_idx = th_ivech(torch.arange(covh.data.size()[-1]))
        diag_idx = Variable(diag_idx.data.diag().long())
        ldet0 = 2*D_star*torch.sum(torch.log(torch.index_select(covh,-1,diag_idx).abs()))
        
        KLD = 0.5 * ( tr + quad - b_sz*self.N*D_star + ldet1 - ldet0 )
        # Normalise by same number of elements as in reconstruction
        KLD /= b_sz
        return KLD
    
    def _nll_loss(self, mean, logcov, x): 
        # log det (covh) + 0.5 (x-mu)' cov^(-1) (x-mu)
        # take one sample, reconstruction loss
#         criterion = nn.MSELoss()
#         NLL = criterion(mean, x)
        NLL= 0.5 * torch.sum( logcov + 1.0/logcov.exp() * (x-mean).pow(2) + np.log(2*np.pi) )
        
        b_sz = mean.size()[0]
        NLL /= b_sz
        return NLL


    def _bce_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim))
        return BCE









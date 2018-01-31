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
from util.batchutil import *


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, t_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.t_dim = t_dim
        d_dim = int(x_dim/t_dim)
        self.d_dim = d_dim
        l_dim = int(d_dim * (d_dim+1)/2)
        self.l_dim = l_dim
        z_dim = t_dim * l_dim
        self.z_dim = z_dim

        # feature
        self.fc0 = nn.Linear(x_dim, h_dim)
        # encode
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        # transform
        self.fc3 = nn.Linear(z_dim, h_dim)
        # decode
        self.fc41 = nn.Linear(h_dim, x_dim)
        self.fc42 = nn.Linear(h_dim, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # t = torch.linspace(0,2,steps=t_dim+1); t = t[1:]
        # self.K = Variable(torch.exp(-torch.pow(t.unsqueeze(1)-t.unsqueeze(0),2)/2/2) + 1e-4*torch.eye(t_dim))
        a = torch.randn(t_dim,t_dim)
        self.K = torch.mm(a, a.t()) # make symmetric positive definite
        self.Kh = torch.potrf(self.K)
        self.iK = torch.potri(self.Kh)

    def encode(self, x):
        """ p(z|x)
            C = D + uu'
        """
        h1 = self.relu(self.fc0(x))
        enc_mean = self.fc21(h1)
        enc_cov = self.fc22(h1)
        return enc_mean, enc_cov

    def reparameterize(self, mu, logcov):
        #  mu + R*esp (C = R'R)
        if self.training:
            cov = logcov.mul(0.5).exp_()
            eps = Variable(cov.data.new(cov.size()[0:2]).normal_())
            z = cov.mul(eps).add_(mu)
            return z
        else:
            return mu

    def decode(self, z):
        # p(x|z)~ N(f(z), \sigma )
        h3 = self.relu(self.fc3(z))
        dec_mean = self.fc41(h3)
        dec_cov = self.fc42(h3)
        return dec_mean, dec_cov

    def forward(self, x, dist):
        # encoder 
        enc_mean, enc_cov = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_cov)

        # decoder
        dec_mean, dec_cov = self.decode(z)

        kld_loss = self._kld_loss(enc_mean, enc_cov)
        if dist == "gauss":
            nll_loss = self._nll_loss(dec_mean, dec_cov, x)
        elif dist == "bce":
            nll_loss = self._bce_loss(dec_mean, x)

        return kld_loss, nll_loss, (enc_mean, enc_cov), (dec_mean, dec_cov)

    def sample_z(self, x):
        # encoder 
        enc_mean, enc_cov = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize_lt(enc_mean, enc_cov)
        return z

    def sample_x(self):
        means = []
        covs = []
        
        z = Variable(torch.zeros(128, self.z_dim).normal_())        
        dec_mean, dec_cov = self.decode(z)
        return dec_mean.data.view(dec_mean.size()[0], 1, 28, 28),

    def _kld_loss(self, mu, logcov):
        # q(z|x)||p(z), q~N(mu1,S1), p~N(mu2,S2), mu1=mu, S1=cov, mu2=0, S2=I
        # 0.5 * (log 1 - log prod(cov) -d + sum(cov) + mu^2)
        KLD = 0.5 * torch.sum( -logcov -1 + logcov.exp()+ mu.pow(2))
        # Normalise by same number of elements as in reconstruction
        batch_size = mu.size()[0]
        KLD /= batch_size * self.x_dim
        return KLD

    def _nll_loss(self, mean, logcov, x): 
        # 0.5 * log det (x) + mu s
        # take one sample, reconstruction loss
        # criterion = nn.MSELoss()
        # NLL = criterion(mean, x)
        NLL= 0.5 * torch.sum( logcov + 1.0/logcov.exp() * (x-mean).pow(2) + np.log(2*np.pi) )
        batch_size = mean.size()[0]
        NLL /= batch_size
        return NLL


    def _bce_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim))
        return BCE









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
        self.fc22 = nn.Linear(h_dim, int(t_dim*(t_dim+1)/2))
        # transform
        self.fc3 = nn.Linear(z_dim, h_dim)
        # decode
        self.fc41 = nn.Linear(h_dim, x_dim)
        self.fc42 = nn.Linear(h_dim, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        t = torch.linspace(0,2,steps=t_dim+1); t = t[1:]
        self.K = Variable(torch.exp(-torch.pow(t.unsqueeze(1)-t.unsqueeze(0),2)/2/2) + 1e-4*torch.eye(t_dim))
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

    def reparameterize_lt(self, mu, covh):
        #  re-paremterize latent dist
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
        dec_mean = self.sigmoid(self.fc41(h3))
        dec_cov = self.sigmoid(self.fc42(h3))
        return dec_mean, dec_cov

    def forward(self, x, dist):
        # encoder 
        enc_mean, enc_cov = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize_lt(enc_mean, enc_cov)

        # decoder
        dec_mean, dec_cov = self.decode(z)

        kld_loss = self._kld_loss_bkdg(enc_mean, enc_cov)
        if dist == "gauss":
            nll_loss = self._nll_loss(dec_mean, x)
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


    def _kld_loss_bkdg(self, mu, covh):
        # q(z|x)||p(z), q~N(mu0,S0), p~N(mu1,S1), mu0=mu, S0=cov, mu1=0, S1=I
        # KLD = 0.5 * ( log det(S1) - log det(S0) -D + trace(S1^-1 S0) + (mu1-mu0)^TS1^-1(mu1-mu0) )
        
        cov = bivech2(covh)
        tr = self.l_dim*torch.sum(torch.mul(self.iK,cov))
        vechI = Variable(th_vech(torch.eye(self.d_dim))).view(1,1,-1)
        b_sz = mu.size()[0]
        I_mu = vechI - mu.view(b_sz,self.t_dim,-1)
        quad = torch.sum( torch.mul(torch.matmul(self.iK,I_mu), I_mu) )
        ldet1 = 2*b_sz*self.l_dim*torch.sum(torch.log(self.Kh.diag().abs()))
    #         diag_idx = th_ivech(torch.arange(covh.data.size()[-1])).diag().long()
        diag_idx = th_ivech(torch.arange(covh.data.size()[-1]))
        diag_idx = Variable(diag_idx.data.diag().long())
        ldet0 = 2*self.l_dim*torch.sum(torch.log(torch.index_select(covh,-1,diag_idx).abs()))
        
        KLD = 0.5 * ( tr + quad - b_sz*self.t_dim*self.l_dim + ldet1 - ldet0 )
        # Normalise by same number of elements as in reconstruction
        KLD /= b_sz
        return KLD

    def _nll_loss(self, mean, x): 
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









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


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(x_dim, h_dim) #prior
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim*z_dim)
        self.fc3 = nn.Linear(z_dim*z_dim, h_dim)
        self.fc41 = nn.Linear(h_dim, x_dim)
        self.fc42 = nn.Linear(h_dim, x_dim*x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """ Posteriro p(z|x),
        produce a series of zs
        """
        h1 = self.relu(self.fc1(x))
        mu = self.fc21(h1)
        logcov = self.fc22(h1).view(-1,self.z_dim, self.z_dim)
        return mu, logcov

    def reparameterize(self, mu, logcov):
        # TODO: mu + Aesp, how to get A, predict var
        if self.training:
            cov = logcov.exp_()
            eps = Variable(cov.data.new(cov.size()[0:2]).normal_())
            eps_view = eps.unsqueeze(2)
            z = cov.bmm(eps_view).squeeze(2).add(mu)
            return z
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc41(h3)), self.relu(self.fc42(h3))

    def forward(self, x):
        # encoder 
        enc_mean, enc_std = self.encode(x.view(-1, self.x_dim))
        sigma = Variable(torch.zeros(self.x_dim, self.x_dim))
        for i in range(self.x_dim):
            # sample independent z
            z = self.reparameterize(enc_mean, enc_std)
            sigma = sigma + torch.bmm(z.unsqueeze(2), z.unsqueeze(1))

        # decoder
        dec_mean, dec_std = self.decode(sigma.view(-1, self.x_dim*self.x_dim))

        # loss
        kld_loss = self._kld_loss(enc_mean, enc_std)
        bce_loss = self._bce_loss(dec_mean, x )

        return kld_loss, bce_loss,  (enc_mean, enc_std), (dec_mean, dec_std)

    def sample(self):
        h = Variable(torch.zeros(1, self.x_dim))
        sigma = Variable(torch.zeros(self.x_dim, self.x_dim))

        # encoder 
        enc_mean, enc_std = self.encode(h)
        for i in range(self.x_dim):
            z = self.reparameterize(enc_mean, enc_std)
            sigma = sigma + torch.bmm(z.unsqueeze(2), z.unsqueeze(1))

        sample = sigma.data.squeeze(0)
        print('sample shape', sample.shape)
    
        return sample

    def _kld_loss(self, mu, logvar, prior_mean, prior_cov ):
        # TODO: need to re-write
        # 0.5 * log det(S2)/det(S1) -d + trace(S2^-1 S1) + (mu2-mu1)S2^-2(mu2-mu1)
        var = logvar.exp()
        cov = var.bmm(var)
        KLD =  torch.log( np.linalg.det(cov.data.numpy()) / np.linalg.det(prior_cov.data.numpy()))
        KLD = KLD +  np.trace(cov.inv().bmm.(prior_cov)) + (mu-prior_mean).bmm.(cov.inv())(mu-prior_mean)

        return -0.5*KLD

    def _nll_loss(self, mean, cov, x): 
        tmp = Variable( mean.size()[0]*torch.log(np.linalg.det(cov.data.numpy())))
        return 0.5 * torch.sum(tmp + 1.0/std.pow(2) * (x-mean).t().mm(cov).inv().mm(x-mean))


    def _bce_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim))
        return BCE

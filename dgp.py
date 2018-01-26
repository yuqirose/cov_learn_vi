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
from util.batchutil import bivech



class DGP(nn.Module):
    """deep gaussian process """
    def __init__(self, x_dim, h_dim, t_dim):
        super(DGP, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.t_dim = t_dim
        d_dim = x_dim/t_dim
        l_dim = int(d_dim*(d_dim+1)/2)
        z_dim = t_dim * l_dim
        self.z_dim = z_dim


        # encode 1: x -> f
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc11 = nn.Linear(h_dim, t_dim)
        self.fc12 = nn.Linear(h_dim, int(t_dim*(t_dim+1)/2))
         
        # encode 2: f -> z
        self.fc2 = nn.Linear(t_dim, h_dim)

        self.fc21 = nn.Linear(h_dim, z_dim) 
        self.fc22 = nn.Linear(h_dim, z_dim)


        # decode
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, x_dim)
        self.fc32 = nn.Linear(h_dim, x_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encode_1(self, x):
        # f(\xi)
        h1 = self.relu(self.fc1(x))
        enc_mean = self.fc11(h1)
        enc_cov = self.fc12(h1)
        return enc_mean, enc_cov

    def encode_2(self, x):
        # L
        h2 = self.relu(self.fc2(x))
        enc_mean = self.fc21(h2)
        enc_cov = self.fc22(h2)
        return enc_mean, enc_cov

    def reparameterize_gp(self, mu, logcov):
        #  z = mu+ Lxi
      
        if self.training:
            b_sz = mu.size()[0]
            cov = logcov.exp_()
            xi = Variable(mu.data.new(mu.size()).normal_()).view(b_sz,self.t_dim,-1)
            covh_sqform = bivech(cov) 
            z = covh_sqform.bmm(xi).view(b_sz,-1).add(mu)
            return z
        else:
            return mu


    def reparameterize_nm(self, mu, logcov):
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
        dec_mean = self.fc31(h3)
        dec_cov = self.fc32(h3)
        return dec_mean, dec_cov

    def forward(self, x, dist):
        # r(f|x)
        va_sz = 100
        xi = torch.distributions.normal(va_sz)
        f_mean, f_cov = self.encode_1(x.view(-1, self.x_dim), xi)
        f = self.reparameterize_gp(f_mean, f_cov)

        # q(z|f)
        z_mean, z_cov = self.encode_2(f)
        z = self.reparameterize_nm(z_mean, z_cov)
    
        # p(x|z)
        x_mean, x_cov = self.decode(z)

        kld_loss = self._kld_loss(z_mean, z_cov) + self.kld_loss_mvn(f_mean, f_cov)
        if dist == "gauss":
            nll_loss = self._nll_loss(x_mean, x_cov, x)
        elif dist == "bce":
            nll_loss = self._bce_loss(x_mean, x)

        nll_loss = self._nll_loss(x_mean, x_cov, x)

        return kld_loss, nll_loss,(f_mean, f_cov), (x_mean, x_cov)

    def sample_z(self, x):
        # encoder 
        f_mean, f_cov = self.encode_1(x)
        f = self.reparameterize_gp(f_mean, f_cov)

        z_mean, z_cov = self.encode_2(f)
        z = self.reparameterize_nm(z_mean, z_cov)
        return z

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


    def _kld_loss_mvn(self, mu, covh):
        # KL loss for multivariate normal 
        # 0.5 * [ log det(S2) - log det(S1) -d + trace(S2^-1 S1) + (mu2-mu1)^TS2^-1(mu2-mu1)]
        b_size = mu.size()[0]
        S1 = bivech2(covh)
        S2 = 


        KLD = batch_trace(S2) - batch_trace(S1) - S1.size()[1]+ batch_trace(torch.bmm(S2,S1))
        mu = mu2-mu1
        cov_inv2 = batch_inverse(cov2)
        KLD = KLD + torch.bmm(torch.bmm(mu.unsqueeze(1), cov_inv2), mu.unsqueeze(2))
        KLD = 0.5 * torch.sum(KLD)
        return KLD

    def _nll_loss(self, mean, covh, x): 
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







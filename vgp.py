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
from util.batchutil import bivech2
# from tensor_util import *



class VGP(nn.Module):
    """variational gaussian process """
    def __init__(self, x_dim, h_dim, t_dim):
        super(VGP, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.t_dim = t_dim
        d_dim = int(x_dim/t_dim)
        self.d_dim = d_dim
        z_dim = x_dim

        # encode 1: x -> s,t (variational data)
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc11 = nn.Linear(h_dim, t_dim)
        self.fc12 = nn.Linear(h_dim, d_dim)
         
        # encode 2: f -> z
        self.fc2 = nn.Linear(t_dim+d_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim) 
        self.fc22 = nn.Linear(h_dim, z_dim)

        # encode 3: x, z -> r
        self.fc23 = nn.Linear(x_dim + z_dim, h_dim)
        self.fc24 = nn.Linear(h_dim, t_dim+d_dim)
        self.fc25 = nn.Linear(h_dim, t_dim+d_dim)

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
        s = self.fc11(h1)
        t = self.fc12(h1)
        return s, t

    def encode_2(self, f, xi):
        # [x, z] - > \xi,f  
        inp = torch.cat((f, xi), 1)
        h2 = self.relu(self.fc2(inp))
        enc_mean = self.fc21(h2)
        enc_cov = self.fc22(h2)
        return enc_mean, enc_cov

    def encode_3(self, x, z):
        inp = torch.cat((x,z), 1)
        h2 = self.relu(self.fc23(inp))
        enc_mean = self.fc24(h2)
        enc_cov = self.fc25(h2)
        xi_mean, fxi_mean = enc_mean.narrow(1, 0, self.t_dim), enc_mean.narrow(1, self.t_dim, self.d_dim)  # slice
        xi_cov, fxi_cov = enc_cov.narrow(1, 0, self.t_dim), enc_cov.narrow(1, self.t_dim, self.d_dim) 
        return xi_mean, fxi_mean, xi_cov, fxi_cov


    def reparameterize_gp(self, xi, s, t):
        #  reparameterize gp dist
        if self.training:
            b_sz = s.size()[0]
            kk_inv = self.kernel(xi,s).mm(self.kernel(s,s).inverse())
            K = self.kernel(s,s)
            mu = kk_inv.unsqueeze(1).matmul(t).squeeze(1)
            cov = self.kernel(xi, xi)-kk_inv.mul(self.kernel(s,xi).transpose(0,1))
            # L, piv = torch.pstrf(cov) #cholesky decomposition
            cov_diag = cov.diag()
            print(cov_diag.data.numpy())

            L = torch.sqrt(cov_diag)
            eps = Variable(t.data.new(t.size()).normal_())
            f = torch.diag(L).matmul(eps).add(mu)
            return f
        else:
            return mu

    def kernel(self, x, y):
        # evaluate kernel value given data pair

        def _ard(x,y, sigma2, w):
            # automatic relavance determination kernel
            return w.dot((x-y).pow(2)).mul(-0.5).exp_().mul(sigma2)
        
        sigma2 = Variable(torch.FloatTensor([1]))
        w = Variable(torch.ones(x.size()[1]))
        b_sz = x.size()[0]
        K = Variable(torch.zeros((b_sz, b_sz)))
     
        for i in range(b_sz):
            for j in range(b_sz):
                K[i,j] = _ard(x[i,],y[j,], sigma2, w)
        return K

    def reparameterize_nm(self, mu, logcov):
        #  reparemterize normal dist
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
        s, t = self.encode_1(x.view(-1, self.x_dim))

        # latent input
        xi = Variable(s.data.new(s.size()).normal_())
        print('xi shape', xi.shape)
        f = self.reparameterize_gp(xi, s, t)

        # q(z|f)
        z_mean, z_cov = self.encode_2(f, xi)
        z = self.reparameterize_nm(z_mean, z_cov)
    
        # r(xi|x, z)
        xi_mean, fxi_mean, xi_cov, fxi_cov  = self.encode_3(x, z)


        # p(x|z)
        x_mean, x_cov = self.decode(z)

        kld_loss = self._kld_loss(z_mean, z_cov)+ self._kld_loss(fxi_mean, fxi_cov)

        if dist == "gauss":
            nll_loss = self._nll_loss(x_mean, x_cov, x)
        elif dist == "bce":
            nll_loss = self._bce_loss(x_mean, x) 

        q_xi = torch.distributions.Normal(torch.zeros(xi.size()), torch.ones(xi.size()))
        log_q_xi =  q_xi.log_prob(xi.data).sum()
        log_r_xi = self._nll_loss(xi_mean, xi_cov, xi)

        nll_loss = nll_loss + log_q_xi - log_r_xi
        return kld_loss, nll_loss,(z_mean, z_cov), (x_mean, x_cov)

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
        S2 = torch.eye(mu.size()[1]).repeat(b_size)

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



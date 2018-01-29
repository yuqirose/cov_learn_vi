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
from tensor_util import *

"""
Modified by
Shiwei Lan @ CalTech, 2018
version 0.1
"""

class VGP(nn.Module):
    """variational gaussian process """
    def __init__(self, x_dim, h_dim, t_dim):
        super(VGP, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.t_dim = t_dim
        d_dim = int(x_dim/t_dim)
        self.d_dim = d_dim
        d2h_dim = int(d_dim*(d_dim+1)/2)
        self.d2h_dim = d2h_dim
        z_dim = t_dim*d2h_dim
        self.z_dim = z_dim
        # source-target dimensions of GP draws
        f_in = t_dim; self.f_in = f_in
        f_out = t_dim; self.f_out = f_out

        # encode 1: x -> (s,t) (variational data)
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc11 = nn.Linear(h_dim, t_dim)
        self.fc12 = nn.Linear(h_dim, t_dim)
         
        # encode 2: f -> z
        self.fc2 = nn.Linear(f_out, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, int(t_dim*(t_dim+1)/2))

        # encode 3: x, z -> r
        self.fc3 = nn.Linear(x_dim + z_dim, h_dim)
        self.fc31 = nn.Linear(h_dim, f_in+f_out)
        self.fc32 = nn.Linear(h_dim, f_in+f_out)

        # decode: z -> x
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc41 = nn.Linear(h_dim, x_dim)
        self.fc42 = nn.Linear(h_dim, x_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # GP kernel
        t = torch.linspace(0,2,steps=t_dim+1); t = t[1:]
        self.K = Variable(torch.exp(-torch.pow(t.unsqueeze(1)-t.unsqueeze(0),2)/2/2) + 1e-4*torch.eye(t_dim))
        self.Kh = torch.potrf(self.K)
#         self.iK = Variable(torch.inverse(self.K.data))
        self.iK = torch.potri(self.Kh)

    def encode_1(self, x):
        # x -> (s,t)
        h1 = self.relu(self.fc1(x))
        s = self.fc11(h1)
        t = self.fc12(h1)
        return s, t

    def encode_2(self, f):
        # f - > z 
        h2 = self.relu(self.fc2(f))
        mean = self.fc21(h2)
        covh = self.fc22(h2)
        return mean, covh

    def encode_3(self, x, z):
        # x, z -> r
        inp = torch.cat((x,z), 1)
        h3 = self.relu(self.fc3(inp))
        enc_mean = self.fc31(h3)
        enc_lcov = self.fc32(h3)
        xi_mean, f_mean = enc_mean.narrow(1, 0, self.f_in), enc_mean.narrow(1, self.f_in, self.f_out)  # slice
        xi_lcov, f_lcov = enc_lcov.narrow(1, 0, self.f_in), enc_lcov.narrow(1, self.f_in, self.f_out) 
        return xi_mean, f_mean, xi_lcov, f_lcov


    def reparameterize_gp(self, xi, s, t):
        #  re-parameterize gp dist
        if self.training:
            b_sz = s.size()[0]
            K_xis = self.kernel(xi,s)
            kk_inv = K_xis.mm(self.kernel(s,s).inverse())
#             K = self.kernel(s,s)
            mu = kk_inv.unsqueeze(1).matmul(t).squeeze(1)
#             cov = self.kernel(xi, xi)-kk_inv.mm(self.kernel(s,xi))
#             # L, piv = torch.pstrf(cov) #cholesky decomposition
#             cov_diag = cov.diag()
            cov_diag = self.kernel(xi, xi).diag() - torch.sum(kk_inv.mul(K_xis),1)
            print(cov_diag.data.numpy())
            L = torch.sqrt(cov_diag)
            eps = Variable(t.data.new(t.size()).normal_())
            f = torch.diag(L).matmul(eps).add(mu)
            return f, mu, cov_diag
        else:
            return mu, mu, cov_diag

    def kernel(self, x, y):
        # evaluate kernel value given data pair

#         def _ard(x,y, sigma2, w):
#             # automatic relavance determination kernel
#             return w.dot((x-y).pow(2)).mul(-0.5).exp_().mul(sigma2)
        _ard = lambda x,y, sigma2, w: w.dot((x-y).pow(2)).mul(-0.5).exp_().mul(sigma2)
        
        sigma2 = Variable(torch.FloatTensor([1]))
        w = Variable(torch.ones(x.size()[1]))
        b_sz = x.size()[0]
        K = Variable(torch.zeros((b_sz, b_sz)))
     
        for i in range(b_sz):
            for j in range(b_sz):
                K[i,j] = _ard(x[i,],y[j,], sigma2, w)
        return K

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
        h4 = self.relu(self.fc4(z))
        dec_mean = self.fc41(h4)
        dec_lcov = self.fc42(h4)
        return dec_mean, dec_lcov

    def forward(self, x, dist):
        # xi
        s, t = self.encode_1(x.view(-1, self.x_dim))
        xi  = Variable(s.data.new(s.size()).normal_())

        # q(f|xi, s, t)
        f, qf_mean, qf_cov = self.reparameterize_gp(xi, s, t)

        # q(z|f)
        z_mean, z_covh = self.encode_2(f)
        z = self.reparameterize_lt(z_mean, z_covh)
    
        # r(xi, f|z, x)
        xi_mean, rf_mean, xi_lcov, rf_lcov  = self.encode_3(x, z)
        
        qf_lcov = qf_cov.log().clone().repeat(rf_lcov.data.size()[1],1).t()
        
        kld_loss = self._kld_loss_bkdg(z_mean, z_covh) + self._kld_loss_diag(qf_mean, qf_lcov, rf_mean, rf_lcov)

        # p(x|z)
        x_mean, x_lcov = self.decode(z)
        
        if dist == "gauss":
            nll_loss = self._nll_loss(x_mean, x_lcov, x)
        elif dist == "bce":
            nll_loss = self._bce_loss(x_mean, x) 

        q_xi = torch.distributions.Normal(torch.zeros(xi.size()), torch.ones(xi.size()))
        log_q_xi =  q_xi.log_prob(xi.data).sum()
        log_r_xi = self._nll_loss(xi_mean, xi_lcov, xi)

        nll_loss = nll_loss + log_q_xi - log_r_xi
        return kld_loss, nll_loss,(z_mean, z_covh), (x_mean, x_lcov)

    def sample_z(self, x):
        # encoder 
        f_mean, f_cov = self.encode_1(x)
        xi = Variable(f_mean.data.new(f_mean.size()).normal_())
        f, _,_ = self.reparameterize_gp(xi, f_mean, f_cov)

        z_mean, z_covh = self.encode_2(f)
        z = self.reparameterize_lt(z_mean, z_covh)
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
    
    def _kld_loss_bkdg(self, mu, covh):
        # q(z|x)||p(z), q~N(mu0,S0), p~N(mu1,S1), mu0=mu, S0=cov, mu1=0, S1=I
        # KLD = 0.5 * ( log det(S1) - log det(S0) -D + trace(S1^-1 S0) + (mu1-mu0)^TS1^-1(mu1-mu0) )
        
        cov = bivech2(covh)
        tr = self.d2h_dim*torch.sum(torch.mul(self.iK,cov))
        vechI = Variable(th_vech(torch.eye(self.d_dim))).view(1,1,-1)
        b_sz = mu.size()[0]
        I_mu = vechI - mu.view(b_sz,self.t_dim,-1)
        quad = torch.sum( torch.mul(torch.matmul(self.iK,I_mu), I_mu) )
        ldet1 = 2*b_sz*self.d2h_dim*torch.sum(torch.log(self.Kh.diag().abs()))
#         diag_idx = th_ivech(torch.arange(covh.data.size()[-1])).diag().long()
        diag_idx = th_ivech(torch.arange(covh.data.size()[-1]))
        diag_idx = Variable(diag_idx.data.diag().long())
        ldet0 = 2*self.d2h_dim*torch.sum(torch.log(torch.index_select(covh,-1,diag_idx).abs()))
        
        KLD = 0.5 * ( tr + quad - b_sz*self.t_dim*self.d2h_dim + ldet1 - ldet0 )
        # Normalise by same number of elements as in reconstruction
        KLD /= b_sz
        return KLD

    def _kld_loss_diag(self, mu0, ls0, mu1, ls1):
        s1_inv = ls1.mul(-1).exp()
        KLD = 0.5 * torch.sum( ls1 -ls0 -1 + s1_inv.mul(ls0.exp())+ (mu0-mu1).pow(2).mul(s1_inv) )
        # Normalise by same number of elements as in reconstruction
        b_sz = mu0.size()[0]
        KLD /= b_sz
        return KLD

    def _kld_loss_mvn(self, mu1, s1, mu2, s2):
        # KL loss for multivariate normal 
        # 0.5 * [ log det(S2) - log det(S1) -d + trace(S2^-1 S1) + (mu2-mu1)^TS2^-1(mu2-mu1)]
        KLD = batch_trace(s2) - batch_trace(s1) - s1.size()[1]+ batch_trace(torch.bmm(s2,s1))
        mu = mu2-mu1
        s2_inv = batch_inverse(s2)
        KLD = KLD + torch.bmm(torch.bmm(mu2.unsqueeze(1)-mu1, s2_inv), mu.unsqueeze(2)-mu1)
        KLD = 0.5 * torch.sum(KLD)
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



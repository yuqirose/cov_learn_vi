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

# class Cholesky(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, a):
#         l = torch.potrf(a, False)
#         ctx.save_for_backward(l)
#         return l
#     @staticmethod
#     def backward(ctx, grad_output):
#         l, = ctx.saved_variables
#         # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}
#         # TODO: ideally, this should use some form of solve triangular instead of inverse...
#         linv =  l.inverse()
#         
#         inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
#         s = torch.mm(linv.t(), torch.mm(inner, linv))
#         # could re-symmetrise 
#         #s = (s+s.t())/2.0
#         
#         return s

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        # feature
        self.fc0 = nn.Linear(x_dim, h_dim)
        # encode
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc23 = nn.Linear(h_dim, z_dim)
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
        t = torch.linspace(1/self.N,2,steps=self.N)
        self.K = Variable(.5*torch.exp(-torch.pow(torch.unsqueeze(t,1)-torch.unsqueeze(t,0),2)/2/5) + 1e-5*torch.eye(self.N))
        self.iK = Variable(torch.inverse(self.K.data))

    def encode(self, x):
        """ p(z|x)
            C = D + uu'
        """
        h1 = self.relu(self.fc0(x))
        enc_mean = self.fc21(h1)
        enc_cov = self.fc22(h1)
        #enc_cov_2 = self.fc23(h1)
        return enc_mean, enc_cov

    def reparameterize(self, mu, logcov):
        #  mu + R*esp (C = R'R)
        if self.training:
            logcov/=2
            cov = logcov.exp_()
            eps = Variable(cov.data.new(cov.size()[0:2]).normal_())
            z = cov.mul(eps).add_(mu)
            # eps_view = eps.unsqueeze(2)
            # z = cov.bmm(eps_view).squeeze(2).add(mu)
            return z
        else:
            return mu

    def decode(self, z):
        # p(x|z)~ N(f(z), \sigma )
        # TODO: try hard decode -- without nn
        h3 = self.relu(self.fc3(z))
        dec_mean = self.fc41(h3)
        dec_cov = self.fc42(h3)
        return dec_mean, dec_cov

    def forward(self, x):
        # encoder 
        enc_mean, enc_cov = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_cov)

        # decoder
        dec_mean, dec_cov = self.decode(z)

        kld_loss = self._kld_loss(enc_mean, enc_cov)
        nll_loss = self._nll_loss(dec_mean, dec_cov, x)

        return kld_loss, nll_loss,(enc_mean, enc_cov), (dec_mean, dec_cov)

    def sample_z(self, x):
        # encoder 
        enc_mean, enc_cov = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_cov)
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
#         KLD = 0.5 * torch.sum( -logcov -1 + logcov.exp()+ mu.pow(2))
        batch_size = mu.size()[0]
#         vechI = torch.from_numpy(vech(np.eye(self.D,dtype=np.float32),'row')[None,None,:])
        vechI = torch.eye(self.D)
        vechI = vechI[tril_mask(vechI)].view(1,1,-1)
        mu_I = Variable(mu.view(batch_size,self.N,-1).data - vechI)
        cov = logcov.exp().view(batch_size,self.N,-1)
        D_star = np.int(self.D*(self.D+1)/2)
        KLD = 0.5 * ( torch.sum(torch.matmul(self.iK.diag(),cov)) + torch.sum( torch.mul(torch.matmul(self.iK,mu_I), mu_I) ) \
                        - batch_size*self.N*D_star + 2*batch_size*D_star*torch.sum(torch.log(torch.potrf(self.K).diag().abs())) -torch.sum(logcov) )
        # Normalise by same number of elements as in reconstruction
        KLD /= batch_size
        return KLD
    
    def _nll_loss(self, mean, cov, x): 
        # 0.5 * log det (x) + mu s
        # take one sample, reconstruction loss
        # print('mean', mean)
        # print('x', x)
        criterion = nn.MSELoss()
 
        NLL = criterion(mean, x)
        # NLL= 0.5 * torch.sum( mean.size()[1]*logcov + 1.0/logcov.exp() * (x-mean).pow(2))
        # TODO: hard decode x
#         NLL = 0;
#         for n in range(z.shape[0]):
#             z_n = ivech(z[n,:])
#             NLL += np.sum(np.log(np.abs(np.diag(z_n)))) + .5* np.sum(np.solve(z_n, x.T).faltten())
        
        batch_size = mean.size()[0]
        NLL /= batch_size
        return NLL


    def _bce_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim))
        return BCE









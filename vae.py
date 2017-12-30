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
        self.fc42 = nn.Linear(h_dim, x_dim*x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
            cov = logcov.exp_()
            eps = Variable(cov.data.new(cov.size()[0:2]).normal_())
            z = cov.mul(eps).add_(mu)
            # eps_view = eps.unsqueeze(2)
            # z = cov.bmm(eps_view).squeeze(2).add(mu)
            return z
        else:
            return mu

    def decode(self, z):
        # p(x|z)
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc41(h3)), self.relu(self.fc42(h3))

    def forward(self, x):
        # encoder 
        enc_mean, enc_cov = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_cov)

        # decoder
        dec_mean, dec_cov = self.decode(z)

        # loss
        kld_loss = self._kld_loss(enc_mean, enc_cov)
        bce_loss = self._bce_loss(dec_mean, x )

        return kld_loss, bce_loss,  (enc_mean, enc_cov), (dec_mean, dec_cov)

    def sample(self):
        h = Variable(torch.zeros(1, self.x_dim))
        # encoder 
        enc_mean, enc_cov = self.encode(h)
        z = self.reparameterize(enc_mean, enc_cov)
        sample = z.data.squeeze(0)
        print('sample shape', sample.shape)
        return sample

    def _kld_loss(self, mu, logcov):
        KLD = -0.5 * torch.sum(1 + logcov - mu.pow(2) - logcov.exp())
        # Normalise by same number of elements as in reconstruction
        # KLD /= args.batch_size
        return KLD

    # def _kld_loss(self, mu, logcov):
    #     # prior is normal 
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
        tmp = Variable( mean.size()[0]*torch.log(np.linalg.det(cov.data.numpy())))
        return 0.5 * torch.sum(tmp + 1.0/std.pow(2) * (x-mean).t().mm(cov).inv().mm(x-mean))


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






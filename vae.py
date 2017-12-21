from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """Summary
        
        Args:
            x (x_dim x 1): input 
        
        Returns:
            mu: mean
            var: log variance
        """
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        # encoder 
        enc_mean, enc_std = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(enc_mean, enc_std)

        # decoder
        dec_mean = self.decode(z)

        # loss
        kld_loss = self._kld_loss(enc_mean, enc_std)
        bce_loss = self._bce_loss(dec_mean, x)

        return kld_loss, bce_loss,  (enc_mean, enc_std), dec_mean

    def _kld_loss(self, mu, logvar):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD = torch.mean(KLD)
        return KLD

    def _bce_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.x_dim))
        return BCE

import math
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt 
import seaborn as sns
from vae import VAE
from dgp import DGP
from reader import SynthDataset
from plot import plot_ts
import visdom 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

def train(epoch):
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):

        #transforming data
        data = Variable(data)
        # print('data shape', data[0,].shape)

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, (enc_mean, enc_cov), (dec_mean, dec_cov) = model(data, 'bce')
        
        # loss
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm(model.parameters(), clip)

        #printing
        if batch_idx % args.print_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.4f} \t NLL Loss: {:.4f} \t ELBO Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                kld_loss.data[0] / args.batch_size,
                nll_loss.data[0] / args.batch_size,
                loss.data[0] /args.batch_size))

        train_loss += loss.data[0]


    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

   # plot the data and reconstruction
    if is_plot:
        plot_img()

def test(epoch):
    """uses test data to evaluate 
    likelihood of the model"""
    
    mean_kld_loss, mean_nll_loss = 0, 0
    for i, (data, _) in enumerate(test_loader):                                            
        
        data = Variable(data)
        # data = Variable(data.squeeze().transpose(0, 1))
        # data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])

        kld_loss, nll_loss,(enc_mean, enc_cov), (dec_mean, dec_cov) = model(data, 'bce')
        mean_kld_loss += kld_loss.data[0]
        mean_nll_loss += nll_loss.data[0]

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)

    print('====> Test set loss: KLD Loss: {:.4f}, NLL Loss: {:.4f}'.format(
        mean_kld_loss, mean_nll_loss))

  # log-odds
    ll = torch.distributions.Normal(dec_mean, 1.0)
    nll = -torch.sum(ll.log_prob(data.view(-1, x_dim)))/(args.batch_size)
    print('test log likelihood', nll.data)






#hyperparameters
N = 200 
D = 2
x_dim = N*D #28*28 
h_dim = 100
z_dim = np.int(N*D*(D+1)/2) #20
n_layers =  1
clip = 1.10
is_plot=True
data_set = "synth"


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before printing training status')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#init model + optimizer + datasets
# SynthDataset(train=True)
# datasets.MNIST('data', train=True, download=True,
#   transform=transforms.ToTensor())
if data_set == "synth":
    train_loader = torch.utils.data.DataLoader(
        SynthDataset(train=True),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        SynthDataset(train=False),
        batch_size=args.batch_size, shuffle=True)

elif data_set == "mnist":   
    train_loader = torch.utils.data.DataLoader(
       datasets.MNIST('data', train=True, download=True,
  transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
       datasets.MNIST('data', train=False, download=True,
  transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)



model = VAE(x_dim, h_dim, z_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, args.epochs + 1):
    
    #training + testing
    train(epoch)
    test(epoch)

    # save_image(sample.data.view(64, 1, 28, 28),
        # 'results/sample_' + str(epoch) + '.png')

    #saving model

    if epoch % args.log_interval == 1:
        fn = 'saves/vae_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)









"""
Plot funcs 
Jan, 2018 Rose Yu @Caltech 
"""
import matplotlib.pyplot as plt 
import seaborn as sns
from util.matutil import ivech2x, vechx

def plot_img():
    """ 
    plot ground truth (left) and reconstruction (right)
    showing b/w image data of mnist
    """
    plt.subplot(121)
    plt.imshow(data.data.numpy()[0,].squeeze())
    plt.subplot(122)
    plt.imshow(dec_mean.view(-1,28,28).data.numpy()[0,].squeeze())

    plt.show()
    plt.pause(1e-6)
    plt.gcf().clear()
    sample = model.sample_z(data)    
    plt.imshow(sample)

def plot_kde():
    """
    plot the kernel density estimation for 2d distributions
    """
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    sns.kdeplot(data.data.numpy()[:,0], data.data.numpy()[:,1], color="r", shade=True, ax=ax1)
    sns.kdeplot(dec_mean.data.numpy()[:,0], dec_mean.data.numpy()[:,1], color="b", shade=True, ax=ax2)
    plt.show()
    plt.pause(1e-6)
    plt.gcf().clear()

def plot_ts(data, enc, dec):
    """
    plot time series with uncertainty
    """
    enc_mean, enc_cov = enc
    dec_mean, dec_cov = dec

    batch_size = enc_mean.size()[0]
    N = 200
    D = 2

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, sharex=True)
    # plot data
    plt.axes(ax1)
    sns.tsplot(data.view(batch_size,N,-1).data.numpy())

    # plot reconstruction
    plt.axes(ax2)
    
    # plot latent variables
    # sample_L = model.sample_z(data).reshape((batch_size,N,-1))
    sample_L = enc_mean.view(batch_size,N,-1).data.numpy()
    sample_Sigma = ivech2x(sample_L)
    sample_vechSigma = vechx(sample_Sigma.reshape((-1,D,D))).reshape((batch_size,N,-1))
    sns.tsplot(sample_vechSigma)


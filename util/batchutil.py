#!/usr/bin/env python
"""
Batch utilities for matrix transformation
Shiwei Lan @ Caltech, Jan. 2018
version 0.2
"""

import numpy as np
import torch
from torch.autograd import Variable
import warnings

def th_vech(L, order='row'):
    """ Gauss vech function """
    "vectorize a lower triangular matrix in the chosen order"
    D = L.size()[0]
    
    if 'row' in order:
        ind = torch.tril(torch.ones(D,D)) == 1
        v = L[ind]
    elif 'col' in order:
        ind = torch.triu(torch.ones(D,D)) == 1
        R = L.t()
        v = R[ind]
    else:
        warings.warn('Wrong order!','once')
        v = torch.Tensor(np.int(D*(D+1)/2))
        v[:] = float('NaN')
    
    return v

def bvech(L, order='row'):
    """ batch Gauss vech function """
    "vectorize a lower triangular matrix in the chosen order"
    "extended to time series data"
    s = L.size() # L : (m,N,D,D)
    D = s[-1]; D_star = np.int(D*(D+1)/2)
    
    if 'row' in order:
        ind = Variable(torch.tril(torch.ones(D,D)) == 1)
        v = torch.masked_select(L.view(-1,D,D),ind.unsqueeze(0))
        v = v.view(s[:-2]+(-1,)) # (m,N,D_star)
    elif 'col' in order:
        ind = Variable(torch.triu(torch.ones(D,D)) == 1)
        R = L.transpose(-1,-2).contiguous()
        v = torch.masked_select(R.view(-1,D,D),ind.unsqueeze(0))
        v = v.view(s[:-2]+(-1,))
    else:
        warings.warn('Wrong order!','once')
        v = torch.Tensor(s[:-2]+(D_star,))
        v[:] = float('NaN')
    
    return v

def th_ivech(v, order='row'):
    """ inverse Gauss vech function """
    "restore the lower triangular matrix from a vector with the chosen order"
    l = v.numel()
    D = np.int(np.ceil((np.sqrt(1+8*l)-1)/2))
    
    L = Variable(torch.zeros(D,D))
    if 'row' in order:
        ind = torch.tril(torch.ones(D,D)) == 1
        L[ind] = v
    elif 'col' in order:
        ind = torch.triu(torch.ones(D,D)) == 1
        L[ind] = v
        L = L.t()
    else:
        warings.warn('Wrong order!','once')
    
    return L

def bivech(v, order='row'):
    """ batch inverse Gauss vech function """
    "restore the lower triangular matrix from a vector with the chosen order"
    "extended to time series data"
    s = v.size() # v : (m,N,D_star)
    l = s[-1]
    D = np.int(np.ceil((np.sqrt(1+8*l)-1)/2))
    
    L = Variable(torch.zeros(np.int(np.prod(s[:-1])),D**2))
    if 'row' in order:
        ind = torch.tril(torch.ones(D,D)) == 1
        idx = torch.nonzero(ind.view(-1))
        L[:,idx] = v.view(-1,l)
        L = L.view(s[:-1]+(D,D)) # (m,N,D,D)
    elif 'col' in order:
        ind = torch.triu(torch.ones(D,D)) == 1
        idx = torch.nonzero(ind.view(-1))
        L[:,idx] = v.view(-1,l)
        L = L.view(s[:-1]+(D,D))
        L = L.transpose(-1,-2).contiguous()
    else:
        warings.warn('Wrong order!','once')
    
    return L

def bivech2(vechL, order='row'):
    """ inverse Gaussian vech2 function """
    "restore the matrix Sigma=LL' from vechL with the chosen order"
    "extended to time series data"
    s = vechL.size() # (m,N,D_star)
    
    L = bivech(vechL,order); 
#     L = L.view((-1,)+L.size()[-2:])
#     Lt = L.transpose(-1,-2)
#     Sigma = torch.bmm(L,Lt)
#     Sigma = Sigma.view(s[:-1]+Sigma.size()[-2:])
    Sigma = torch.matmul(L, L.transpose(-1,-2)) # (m,N,D,D)
    
    return Sigma

def tril_mask(value):
    n = value.size(-1)
    coords = value.new(n)
    torch.arange(n, out=coords)
    return coords <= coords.view(n, 1)


if __name__=='__main__':
    torch.manual_seed(2018)
    D=4; D_star=np.int(D*(D+1)/2);
    
    v=Variable(torch.randn(D_star))
    print(v.unsqueeze(1).t())
    L=th_ivech(v)
    print(L)
    vv=th_vech(L,'col')
    print(vv)
    
    m=2; N=3
    v=Variable(torch.randn(m,N,D_star))
    print(v)
    L=bivech(v)
    print(L)
    vv=bvech(L,'col')
    print(vv)
    Sigma=bivech2(vv)
    print(Sigma)
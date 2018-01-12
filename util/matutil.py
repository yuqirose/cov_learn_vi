#!/usr/bin/env python
"""
Functions for matrix transformation
Shiwei Lan @ Caltech, Jan. 2018
"""

import numpy as np
import warnings

def vech(L,order='row'):
    """ Gauss vech function """
    "vectorize a lower triangular matrix in the chosen order"
    D=L.shape[0]
    
    if 'row' in order:
        ind=np.tril(np.ones((D,D), dtype=bool))
        v=L[ind]
    elif 'col' in order:
        ind=np.triu(np.ones((D,D), dtype=bool))
        R=L.T
        v=R[ind]
    else:
        warings.warn('Wrong order!','once')
        v=np.empty((D*(D+1)/2,))
        v[:]=np.nan
    
    return v

def vechx(L,order='row'):
    """ Gauss vech function """
    "vectorize a lower triangular matrix in the chosen order"
    "extended to time series data"
    D=L.shape[1] # L : (N,D,D)
    
    if 'row' in order:
        ind=np.tril(np.ones((D,D), dtype=bool))
        v=L[:,ind]
    elif 'col' in order:
        ind=np.triu(np.ones((D,D), dtype=bool))
        R=np.swapaxes(L,-1,-2)
        v=R[:,ind]
    else:
        warings.warn('Wrong order!','once')
        v=np.empty((L.shape[0],D*(D+1)/2,))
        v[:]=np.nan
    
    return v

def ivech(v,order='row'):
    """ inverse Gauss vech function """
    "restore the lower triangular matrix from a vector with the chosen order"
    l=len(v)
    D=np.int(np.ceil((np.sqrt(1+8*l)-1)/2))
    
    L=np.zeros((D,D))
    if 'row' in order:
        ind=np.tril(np.ones((D,D), dtype=bool))
        L[ind]=v
    elif 'col' in order:
        ind=np.triu(np.ones((D,D), dtype=bool))
        L[ind]=v
        L=L.T
    else:
        warings.warn('Wrong order!','once')
    
    return L

def ivechx(v,order='row'):
    """ inverse Gauss vech function """
    "restore the lower triangular matrix from a vector with the chosen order"
    "extended to time series data"
    l=v.shape[1]
    D=np.int(np.ceil((np.sqrt(1+8*l)-1)/2))
    
    L=np.zeros((v.shape[0],D,D))
    if 'row' in order:
        ind=np.tril(np.ones((D,D), dtype=bool))
        L[:,ind]=v
    elif 'col' in order:
        ind=np.triu(np.ones((D,D), dtype=bool))
        L[:,ind]=v
        L=np.swapaxes(L,-1,-2)
    else:
        warings.warn('Wrong order!','once')
    
    return L

def ivech2x(vechL,order='row'):
    """ inverse Gaussian vech2 function """
    "restore the matrix Sigma=LL' from vechL with the chosen order"
    "extended to time series data"
    s=vechL.shape
    l=s[-1]
    
    L=ivechx(vechL.reshape((-1,l)),order)
    Sigma=np.zeros_like(L)
    for i in range(Sigma.shape[0]):
        Sigma[i,:,:]=L[i,:,:].dot(L[i,:,:].T)
    Sigma=Sigma.reshape(s[:-1]+Sigma.shape[-2:])
    
    return Sigma


try:
    import torch
except ImportError:
    print('PyTorch not installed!')
    pass
else:
    def tril_mask(value):
        n = value.size(-1)
        coords = value.new(n)
        torch.arange(n, out=coords)
        return coords <= coords.view(n, 1)
    
# alternative:
# def tril_mask(value):
#     try:
#         import torch
#     except ImportError:
#         raise NotImplementedError("PyTorch not installed!")
#     else:
#         n = value.size(-1)
#         coords = value.new(n)
#         torch.arange(n, out=coords)
#         return coords <= coords.view(n, 1)

if __name__=='__main__':
    np.random.seed(2018)
    D=4; D_star=np.int(D*(D+1)/2)
    
    v=np.random.randn(D_star,)
    print(v.T)
    L=ivech(v)
    print(L)
    vv=vech(L,'col')
    print(vv)
    
    N=2
    v=np.random.randn(N,D_star)
    print(v)
    L=ivechx(v)
    print(L)
    vv=vechx(L,'col')
    print(vv)
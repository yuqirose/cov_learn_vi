from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from chol_diff import chol_fwd, chol_rev
import scipy as sp
import numpy as np



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

def batch_diag(X): 
    (batch_sz,dim) = X.size()

    val = Variable(torch.zeros( batch_sz, dim, dim))
    for i in range(X.size()[0]):
        val[i,:,:] =torch.diag(X[i,:])
    return val

# class MyPstrf(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A):
#         ctx.save_for_backward(A)
#         L, piv =  torch.pstrf(A)
#         return L

#     @staticmethod
#     def backward(ctx, dL):
#         A, = ctx.saved_tensors
#         Abar = sp.tril(A)
#         dA = chol_rev(dL, Abar) 
#         return dA


# A = torch.randn(3,3)
# K = Variable(torch.mm(A, A.t()),requires_grad=True)
# pstrf = MyPstrf.apply
# L_hat = pstrf(K)
# L, piv = torch.pstrf(K.data)


# loss = L_hat.pow(2).sum()
# # Use autograd to compute the backward pass.
# loss.backward()
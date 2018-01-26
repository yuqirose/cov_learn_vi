from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from chol_diff import chol_fwd, chol_rev


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

class MyPstrf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, L, Adot):
        return chol_fwd(L, Adot)

    @staticmethod
    def backward(ctx, L, Abar):
        return chol_rev(L, Abar)


A = Variable(torch.randn(3,3))
K = A.mul(A.transpose(0,1))
pstrf = MyPstrf.apply
L = pstrf(K)

loss = (y_pred - y).pow(2).sum()
loss = (y_pred - y).pow(2).sum()
print(t, loss.data[0])

# Use autograd to compute the backward pass.
loss.backward()
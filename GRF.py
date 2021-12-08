from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pandas.core import indexing
from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from scipy.special import gamma
import scipy as sci
from sklearn import gaussian_process as gp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from IPython.display import display
import sys
import h5py
import torch
import math
from math import pi

# from torch._C import device

# import jax
# import jax.numpy as torch
# from jax import grad, jit, vmap, pmap, random
# import jax.scipy as jsci



def construct_points(grid):
    N = grid[0].size
    dim = len(grid)
    R = np.zeros((N, dim))
    for i in range(dim):
        R[:,i] = grid[i].ravel()
    return R

def construct_grid(points,values,shape):
    N = points.shape[0]
    dim = points.shape[1]
    grid = tuple(points[:,i].reshape(shape) for i in range(dim))
    vals = values.reshape(shape)
    return grid, vals


def RBF(x1, x2, output_scale, lengthscales):
    # output_scale, lengthscales = params
    diffs = torch.unsqueeze(x1 / lengthscales, 1) - \
            torch.unsqueeze(x2 / lengthscales, 0)
    r2 = torch.sum(diffs**2, axis=2)
    return output_scale * torch.exp(-0.5 * r2)


def dirichlet_matern(x1, x2=None, Nk=None, l=0.1, sigma=1.0, nu=np.inf, device=None):
    # do even only 
    d = x1.shape[1]
    x1 = x1.to(device)
    if Nk is None:
        Nk = int(math.pow(x1.shape[0],1.0/d))
    # Nk = 2 * (Nk - Nk//2)
    L = x1[-1] - x1[0]
    N_full = torch.tensor([Nk] * d, device=device)
    N_tot = torch.prod(N_full)
    alpha = nu + 0.5*d
    a = torch.sqrt(2.0/L)
    a0 = torch.sqrt(1.0/L)
    kappa = math.sqrt(2*nu)/l
    eta2 = sigma*(4.0*pi)**(0.5*d)*gamma(alpha)/(kappa**d * gamma(nu))
    k = [torch.linspace(1, Nk, Nk, device=device) for _ in range(d)]
    
    k_grid = torch.meshgrid(*k, indexing='ij')
    K = torch.zeros((N_tot, d), device=device)
    for i in range(d):
        K[:, i] = k_grid[i].ravel()
        # K = K.at[:, i].set(k_grid[i].ravel())
    # print(torch.linalg.norm(K, axis=1))
    # Lnorm = torch.linalg.norm(L)
    # Lval = L[0] # this is from coppying the paper, I think we may need to use the Norm of L
    Knorm = torch.linalg.norm(K/L, axis=1)
    eigs_k = 1 + (pi/(kappa))**2 * Knorm**2
    # print(torch.sum(K*x1, axis=1).shape)
    eigs_k = eigs_k[:, None]
    eigs_alpha = eigs_k**(-alpha)
    if np.isinf(nu):
        eigs_k = torch.ones_like(eigs_k)
        eta2 = sigma*(math.sqrt(2.0*pi)*l)**d
        eigs_alpha = torch.exp(-0.5*(l*pi*Knorm)**2)[:, None] # 1/2 instead of 2 for nonperiodic case
    Kx1 = torch.unsqueeze(K,1)*x1
    wk = torch.prod(a*torch.sin(pi/L * Kx1), axis=2)
    if x2 is not None:
        Kx2 = torch.unsqueeze(K,1)*x2
        wk_star = torch.prod(a*torch.sin(pi/L * Kx2), axis=2).T
    else:
        wk_star = wk.T
    
    cov = eta2 * (wk_star @ (wk * eigs_alpha))
    
    return cov
    
    
def neumann_matern(x1, x2=None, Nk=None, l=0.1, sigma=1.0, nu=np.inf, device=None):
    # do even only 
    d = x1.shape[1]
    x1 = x1.to(device)
    L = x1[-1] - x1[0]
    if Nk is None:
        Nk = int(math.pow(x1.shape[0],1.0/d))
    # N = 2 * (N - N//2)
    N_full = torch.tensor([Nk] * d, device=device)
    N_tot = torch.prod(N_full)
    alpha = nu + 0.5*d
    a1 = torch.sqrt(2.0/L)
    a0 = torch.sqrt(1.0/L)
    kappa = math.sqrt(2*nu)/l
    eta2 = sigma*(4.0*pi)**(0.5*d)*gamma(alpha)/(kappa**d * gamma(nu))
    k = [torch.linspace(0, Nk-1, Nk, device=device) for _ in range(d)]
    k_grid = torch.meshgrid(*k, indexing='ij')
    K = torch.zeros((N_tot, d), device=device)
    ones = torch.ones((Nk), device=device)
    a = [ones*a1[i] for i in range(d)]
    # for i in range(d):
    #     a[i][0] = a0[i]
    # a = [a[i].at[0].set(a0[i]) for i in range(d)]
    a_grid = torch.meshgrid(*a, indexing='ij')
    A = torch.zeros((N_tot, d), device=device) # probably inefficient to do it like this, but it nicely keeps track of when k_i is 0 which requires us to use a0 instead of a
    for i in range(d):
        K[:, i] = k_grid[i].ravel()
        A[:, i] = a_grid[i].ravel()
        # K = K.at[:, i].set(k_grid[i].ravel())
        # A = A.at[:, i].set(a_grid[i].ravel())
    # print(torch.linalg.norm(K, axis=1))
    # Lnorm = torch.linalg.norm(L)
    # Lval = L[0] # this is from coppying the paper, I think we may need to use the Norm of L
    Knorm = torch.linalg.norm(K/L, axis=1)
    eigs_k = 1 + (pi/(kappa))**2 * Knorm**2
    # print(torch.sum(K*x1, axis=1).shape)
    eigs_k = eigs_k[:, None]
    eigs_alpha = eigs_k**(-alpha)
    if np.isinf(nu):
        eigs_k = torch.ones_like(eigs_k, device=device)
        eta2 = sigma*(math.sqrt(2.0*pi)*l)**d
        eigs_alpha = torch.exp(-0.5*(l*pi*Knorm)**2)[:, None] # 1/2 instead of 2 for nonperiodic case
    Kx1 = torch.unsqueeze(K,1)*x1
    wk = torch.prod(A*torch.cos(pi/L * Kx1), axis=2)
    if x2 is not None:
        Kx2 = torch.unsqueeze(K,1)*x2
        wk_star = torch.prod(A*torch.cos(pi/L * Kx2), axis=2).T
    else:
        wk_star = wk.T
    
    cov = eta2 * (wk_star @ (wk * eigs_alpha))
    
    return cov
    
    
    
def periodic_matern(x1, x2=None, Nk=None, l=0.1, sigma=1.0, nu=np.inf, device=None):
    d = x1.shape[1]
    x1 = x1.to(device)
    L = x1[-1] - x1[0]
    # L = L.to(device)
    if Nk is None:
        Nk = int(math.pow(x1.shape[0],1.0/d))    # Nk = 2 * (Nk - Nk//2)
    # Nk = 2 * (Nk - Nk//2)
    N_full = torch.tensor([Nk] * d, device=device)
    N_tot = torch.prod(N_full)
    alpha = nu + 0.5*d
    a1 = torch.sqrt(2.0/L)
    a0 = torch.sqrt(1.0/L)
    kappa = math.sqrt(2*nu)/l
    
    eta2 = sigma*(4.0*pi)**(0.5*d)*gamma(alpha)/(kappa**d * gamma(nu))
    k = [torch.linspace(0, Nk-1, Nk, device=device) for _ in range(d)]
    k_grid = torch.meshgrid(*k, indexing='ij')
    
    K = torch.zeros((N_tot, d), device=device)
    ones = torch.ones((Nk), device=device)
    a = [ones*a1[i] for i in range(d)]
    for i in range(d):
        a[i][0] = a0[i]
    # a = [a[i].at[0].set(a0[i]) for i in range(d)]
    a_grid = torch.meshgrid(*a, indexing='ij')
    A = torch.zeros((N_tot, d), device=device) # probably inefficient to do it like this, but it nicely keeps track of when k_i is 0 which requires us to use a0 instead of a
    for i in range(d):
        K[:, i] = k_grid[i].ravel()
        A[:, i] = a_grid[i].ravel()
        # K = K.at[:, i].set(k_grid[i].ravel())
        # A = A.at[:, i].set(a_grid[i].ravel())
    # print(torch.linalg.norm(K, axis=1))
    # Lnorm = torch.linalg.norm(L)
    # Lval = L[0] # this is from coppying the paper, I think we may need to use the Norm of L
    Knorm = torch.linalg.norm(K/L, axis=1)
    eigs_k = 1 + (2.0*pi/(kappa))**2 * Knorm**2
    eigs_k = eigs_k[:, None]
    eigs_alpha = eigs_k**(-alpha)
    if np.isinf(nu):
        eigs_k = torch.ones_like(eigs_k, device=device)
        eta2 = sigma*(math.sqrt(2.0*pi)*l)**d
        eigs_alpha = torch.exp(-2.0*(l*pi*Knorm)**2)[:, None]
        # display(eigs_alpha.shape)
    Kx1 = torch.unsqueeze(K,1)*x1
    wk = torch.prod(a1*(torch.cos(2*pi/L * Kx1) + torch.sin(2*pi/L * Kx1)), axis=2)
    if x2 is not None:
        Kx2 = torch.unsqueeze(K,1)*x2
        wk_star = torch.prod(A*(torch.cos(2*pi/L * Kx2) + torch.sin(2*pi/L * Kx2)), axis=2).T
    else:
        wk_star = wk.T
    
    cov = eta2 * (wk_star @ (wk * eigs_alpha))
    
    return cov
    
    
def get_cholesky(K, jitter=1e-12, device=None):
    N = K.shape[0]
    L = torch.linalg.cholesky(K + jitter*torch.eye(N, device=device))
    return L

def setup_kernel(N, dim, device=None):
    # dim = 2
    # N = 100
    N_full = torch.tensor([N]*dim, device=device)
    Ntot = torch.prod(N_full)
    x = [torch.linspace(0, 1, N, device=device) for i in range(dim)]
    grid = torch.meshgrid(*x, indexing='ij')
    X = torch.zeros((Ntot, dim), device=device, dtype=torch.float64)
    for i in range(dim):
        X[:,i] = grid[i].ravel()
        # X = X.at[:,i].set(grid[i].ravel())
    return X

def generate_sample(L, device=None):
    N = L.shape[0]
    rand = torch.randn((N, 1), dtype=torch.float64, device=device)
    sample = L @ rand
    # sample = torch.dot(L, rand).T
    # sample = np.dot(L, rand).T
    sample = sample.flatten()
    return sample


def generate_samples(L, Nsamples, device=None):
    samples = torch.stack([generate_sample(L, device=device) for N in range(Nsamples)])
    return samples

def plot_sample(X, sample, dim, shape):
    fig = plt.figure()
    if dim == 2:
        X = X.cpu().numpy()
        sample = sample.cpu().numpy()
        grid, U = construct_grid(X, sample, shape=shape)
        X1, X2 = grid
        # X1 = X1.cpu().numpy()
        # X2 = X2.cpu().numpy()
        # U = U.cpu().numpy()
        c = plt.pcolormesh(X1, X2, U, cmap='jet', shading='gouraud', vmin=-2, vmax=2)
        fig.colorbar(c)
        plt.title('GRF Pytorch')
        plt.axis('square')
    elif dim == 1:
        U = sample.cpu().numpy()
        X = X.cpu().numpy()
        plt.plot(X, U)
    plt.show()
    return U

if __name__ == "__main__":
    dim = 1
    N = 101
    l = 0.1
    Nk = None
    Nsamples = 10
    jitter = 1e-12
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    shape = [N]*dim
    X = setup_kernel(N, dim)
    print(f"X={X}")
    K = periodic_matern(X, Nk=Nk, l=l, nu=np.inf, device=device)
    print(f"K={K}")
    L = get_cholesky(K, jitter, device=device)
    print(f"L={L}")
    samples = generate_samples(L, Nsamples, device=device)
    print(samples.shape)
    plt.close('all')
    U = torch.stack([plot_sample(X, sample, dim, shape) for sample in samples])

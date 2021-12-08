from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import os
# import jax
# import jax.numpy as jnp
import numpy as np
import torch
# from jax import random, grad, vmap, jit, hessian, value_and_grad
# from jax.experimental import optimizers
# from jax.experimental.optimizers import adam, exponential_decay
# from jax.experimental.ode import odeint
# from jax.nn import relu, elu, softplus
# from jax.config import config
# # from jax.ops import index_update, index
# from jax import lax
# from jax.lax import while_loop, scan, cond, fori_loop
# from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy
import scipy.io
from scipy.io import loadmat

import sys
import h5py


class BurgersEq1D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                #  ymin=0,
                #  ymax=1,
                #  dx=0.01,
                #  dy=0.01,
                 Nx=100,
                #  Ny=100,
                 nu=0.01,
                 dt=1e-3,
                 tend=1.0,
                 device=None,
                 dtype=torch.float64,
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
        self.x = x
        # self.y = y
        self.dx = x[1] - x[0]
        # self.dy = y[1] - y[0]
        # self.X, self.Y = jnp.meshgrid(x,y,indexing='ij')
        self.nu = nu
        self.u = torch.zeros_like(x, device=device)
        self.u0 = torch.zeros_like(self.u, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.U = []
        self.T = []
        self.device = device
        
    

    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx
    
    # def Dy(self, data):
    #     data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
    #     return data_dy

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    # def Dyy(self, data):
    #     data_dyy = self.CD_ii(data, axis=1, dx=self.dy)
    #     return data_dyy
    

    


    def burgers_calc_RHS(self, u):
        u_xx = self.Dxx(u)
        u2 = u**2.0
        u2_x = self.Dx(u2)
        u_RHS = -0.5*u2_x + self.nu*u_xx
        return u_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def burgers_rk4(self, u, t=0):
        u_RHS1 = self.burgers_calc_RHS(u)
        t1 = t + 0.5*self.dt
#         display(u.shape)
#         display(u_RHS1.shape)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        
        u_RHS2 = self.burgers_calc_RHS(u1)
        t2 = t + 0.5*self.dt
        u2 = self.update_field(u, u_RHS2, step_frac=0.5)
        
        u_RHS3 = self.burgers_calc_RHS(u2)
        t3 = t + self.dt
        u3 = self.update_field(u, u_RHS3, step_frac=1.0)
        
        u_RHS4 = self.burgers_calc_RHS(u3)
        
        t_new = t + self.dt
        u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
        
        return u_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        plt.plot(self.x, self.u)
        # c = plt.pcolormesh(self.X, self.Y, self.u, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        # fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.axis('equal')
        # plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()

        
    def burgers_driver(self, u0, save_interval=10, plot_interval=0):
        # plot results
        # t,it = get_time(time)
#         display(u0[:self.Nx,:self.Ny].shape)
        self.u0 = u0[:self.Nx]
        self.u = self.u0
        self.t = 0
        self.it = 0
        self.T = []
        self.U = []
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.U.append(self.u)
            # self.Psi.append(self.psi)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
#             print(f"t:\t{self.t}")
            self.u, self.t = self.burgers_rk4(self.u, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.U.append(self.u)
                # self.Psi.append(self.psi)
                self.T.append(self.t)

        return torch.stack(self.U)


class BurgersEq2D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 ymin=0,
                 ymax=1,
                #  dx=0.01,
                #  dy=0.01,
                 Nx=100,
                 Ny=100,
                 nu=0.01,
                 dt=1e-3,
                 tend=1.0,
                 device=None,
                 dtype=torch.float64,
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        x = torch.linspace(xmin, xmax, Nx + 1, device=device, dtype=dtype)[:-1]
        y = torch.linspace(xmin, xmax, Ny + 1, device=device, dtype=dtype)[:-1]
        self.x = x
        self.y = y
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.nu = nu
        self.u = torch.zeros_like(self.X, device=device)
        self.u0 = torch.zeros_like(self.u, device=device)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.U = []
        self.T = []
        self.device = device
        
    

    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx
    
    def Dy(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
        return data_dy

    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def Dyy(self, data):
        data_dyy = self.CD_ii(data, axis=1, dx=self.dy)
        return data_dyy
    

    


    def burgers_calc_RHS(self, u):
        u_xx = self.Dxx(u)
        u_yy = self.Dyy(u)
        u2 = u**2.0
        u2_x = self.Dx(u2)
        u2_y = self.Dy(u2)
        u_RHS = -0.5*(u2_x + u2_y) + self.nu*(u_xx + u_yy)
        return u_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def burgers_rk4(self, u, t=0):
        u_RHS1 = self.burgers_calc_RHS(u)
        t1 = t + 0.5*self.dt
#         display(u.shape)
#         display(u_RHS1.shape)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        
        u_RHS2 = self.burgers_calc_RHS(u1)
        t2 = t + 0.5*self.dt
        u2 = self.update_field(u, u_RHS2, step_frac=0.5)
        
        u_RHS3 = self.burgers_calc_RHS(u2)
        t3 = t + self.dt
        u3 = self.update_field(u, u_RHS3, step_frac=1.0)
        
        u_RHS4 = self.burgers_calc_RHS(u3)
        
        t_new = t + self.dt
        u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
        
        return u_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        
        c = plt.pcolormesh(self.X, self.Y, self.u, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        fig.colorbar(c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.axis('square')
        plt.draw() 
        plt.pause(1e-17)
        plt.show()
        
        
    def burgers_driver(self, u0, save_interval=10, plot_interval=0):
        # plot results
        # t,it = get_time(time)
#         display(u0[:self.Nx,:self.Ny].shape)
        self.u0 = u0[:self.Nx]
        self.u = self.u0
        self.t = 0
        self.it = 0
        self.T = []
        self.U = []
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.U.append(self.u)
            # self.Psi.append(self.psi)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
#             print(f"t:\t{self.t}")
            self.u, self.t = self.burgers_rk4(self.u, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.U.append(self.u)
                # self.Psi.append(self.psi)
                self.T.append(self.t)

        return torch.stack(self.U)

    

class BurgersEq3D():
    def __init__(self,
                 xmin=0,
                 xmax=1,
                 ymin=0,
                 ymax=1,
                 zmin=0,
                 zmax=1,
                #  dx=0.01,
                #  dy=0.01,
                #  dz = 0.01,
                 Nx=100,
                 Ny=100,
                 Nz=100,
                 nu=0.0,
                 dt=1e-3,
                 tend=1.0,
                 ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.ymin = zmin
        self.ymax = zmax
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        x = torch.linspace(xmin, xmax, Nx + 1, device=self.device)
        y = torch.linspace(ymin, ymax, Ny + 1, device=self.device)
        z = torch.linspace(zmin, zmax, Nz + 1, device=self.device)
        self.x = x
        self.y = y
        self.z = z
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]
        self.X, self.Y, self.Z = jnp.meshgrid(x,y,z,indexing='ij')
        self.nu = nu
        self.u = jnp.zeros_like(self.X)
        self.psi = jnp.zeros_like(self.X)
        self.u0 = jnp.zeros_like(self.u)
        self.dt = dt
        self.tend = tend
        self.t = 0
        self.it = 0
        self.Phi = []
        self.T = []
        
    

    # All Central Differencing Functions are 4th order.  These are used to compute ann inputs.
    def CD_i(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_i = (data_m2 - 8.0*data_m1 + 8.0*data_p1 - data_p2)/(12.0*dx)
        return data_diff_i

    def CD_ij(self, data, axis_i, axis_j, dx, dy):
        data_diff_i = self.CD_i(data,axis_i,dx)
        data_diff_ij = self.CD_i(data_diff_i,axis_j,dy)
        return data_diff_ij

    def CD_ii(self, data, axis, dx):
        data_m2 = torch.roll(data,shifts=2,dims=axis)
        data_m1 = torch.roll(data,shifts=1,dims=axis)
        data_p1 = torch.roll(data,shifts=-1,dims=axis)
        data_p2 = torch.roll(data,shifts=-2,dims=axis)
        data_diff_ii = (-data_m2 + 16.0*data_m1 - 30.0*data + 16.0*data_p1 -data_p2)/(12.0*dx**2)
        return data_diff_ii

    def Dx(self, data):
        data_dx = self.CD_i(data=data, axis=0, dx=self.dx)
        return data_dx

    def Dy(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dy)
        return data_dy
    
    def Dz(self, data):
        data_dy = self.CD_i(data=data, axis=1, dx=self.dz)
        return data_dy


    def Dxy(self, data):
        data_dxy = self.CD_ij(data, axis_i=0, axis_j=1, dx=self.dx, dy=self.dy)
        return data_dxy

    def Dxz(self, data):
        data_dxz = self.CD_ij(data, axis_i=0, axis_j=1, dx=self.dx, dy=self.dz)
        return data_dxz

    def Dyz(self, data):
        data_dyz = self.CD_ij(data, axis_i=0, axis_j=1, dx=self.dy, dy=self.dz)
        return data_dyz


    def Dxx(self, data):
        data_dxx = self.CD_ii(data, axis=0, dx=self.dx)
        return data_dxx

    def Dyy(self, data):
        data_dyy = self.CD_ii(data,axis=1, dx=self.dy)
        return data_dyy

    def Dzz(self, data):
        data_dzz = self.CD_ii(data,axis=1, dx=self.dz)
        return data_dzz

    
    def burgers_calc_RHS(self, u, psi):
        u_xx = self.Dxx(u)
        u_yy = self.Dyy(u)
        u_zz = self.Dzz(u)
        
        psi_RHS = self.nu * (u_xx + u_yy + u_zz) # it is usually c^2, but c is consistent with simflowny code
        u_RHS = psi
        return u_RHS, psi_RHS
        
    def update_field(self, field, RHS, step_frac):
        field_new = field + self.dt*step_frac*RHS
        return field_new
        

    def rk4_merge_RHS(self, field, RHS1, RHS2, RHS3, RHS4):
        field_new = field + self.dt/6.0*(RHS1 + 2*RHS2 + 2.0*RHS3 + RHS4)
        return field_new

    def burgers_rk4(self, u, psi, t=0):
        u_RHS1, psi_RHS1 = self.burgers_calc_RHS(u, psi)
        t1 = t + 0.5*self.dt
#         display(u.shape)
#         display(u_RHS1.shape)
        u1 = self.update_field(u, u_RHS1, step_frac=0.5)
        psi1 = self.update_field(psi, psi_RHS1, step_frac=0.5)
        
        u_RHS2, psi_RHS2 = self.burgers_calc_RHS(u1, psi1)
        t2 = t + 0.5*self.dt
        u2 = self.update_field(u, u_RHS2, step_frac=0.5)
        psi2 = self.update_field(psi, psi_RHS2, step_frac=0.5)
        
        u_RHS3, psi_RHS3 = self.burgers_calc_RHS(u2, psi2)
        t3 = t + self.dt
        u3 = self.update_field(u, u_RHS3, step_frac=1.0)
        psi3 = self.update_field(psi, psi_RHS3, step_frac=1.0)
        
        u_RHS4, psi_RHS4 = self.burgers_calc_RHS(u3, psi3)
        
        t_new = t + self.dt
        psi_new = self.rk4_merge_RHS(psi, psi_RHS1, psi_RHS2, psi_RHS3, psi_RHS4)
        u_new = self.rk4_merge_RHS(u, u_RHS1, u_RHS2, u_RHS3, u_RHS4)
        
        return u_new, psi_new, t_new
    
    def plot_data(self, cmap='jet', vmin=None, vmax=None, fig_num=0, title='', xlabel='', ylabel=''):
        # plt.ion()
        # fig = plt.figure(fig_num)
        # plt.cla()
        # plt.clf()
        
        # c = plt.pcolormesh(self.X, self.Y, self.u, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        # fig.colorbar(c)
        # plt.title(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.axis('equal')
        # plt.axis('square')
        # plt.draw() 
        # plt.pause(1e-17)
        # plt.show()
        pass

        
    def burgers_driver(self, u0, save_interval=10, plot_interval=0):
        # plot results
        # t,it = get_time(time)
#         display(u0[:self.Nx,:self.Ny].shape)
        self.u0 = u0[:self.Nx,:self.Ny,:self.Nz]
        self.u = self.u0
        
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
        if save_interval != 0 and self.it % save_interval == 0:
            self.Phi.append(self.u)
            # self.Psi.append(self.psi)
            self.T.append(self.t)
        # Compute equations
        while self.t < self.tend:
#             print(f"t:\t{self.t}")
            self.u, self.psi, self.t = self.burgers_rk4(self.u, self.psi, self.t)
            
            self.it += 1
            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(vmin=-1,vmax=1,title=r'\{u}')
            if save_interval != 0 and self.it % save_interval == 0:
                self.Phi.append(self.u)
                # self.Psi.append(self.psi)
                self.T.append(self.t)

        return jnp.array(self.Phi)
    
    

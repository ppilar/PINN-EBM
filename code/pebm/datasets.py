# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy as sp

from .utils_navier_stokes import get_aux

def get_ds(x_opt, dpar=''):
    if type(dpar) != str and type(dpar) != list:
        dpar =  [dpar]
    
    #TODO: fix ds inputs
    if x_opt == 1:
        ds = ds_exp_1d(x_opt, dpar)
    if x_opt == 2:
        ds = ds_sin_1d(x_opt, dpar)
    if x_opt == 3:
        ds = ds_bessel_1d(x_opt, dpar)
    if x_opt == 4:
        ds = ds_sin_2d(x_opt, dpar)
    if x_opt == 5:
        ds = ds_sin_exp_2d(x_opt, dpar)
    if x_opt == 6:
        ds = ds_exp_2d(x_opt, dpar)    
    if x_opt == 101:
        ds = ds_navier_stokes(x_opt)
        
    return ds



class pebm_dataset():
    def __init__(self, x_opt, dpar):
        self.x_opt = x_opt
        self.dpar = dpar
        
    def init_data_ranges(self):  #ranges of input data and normalizing const.
        print('not implemented!')
    
    def init_network_pars(self):  #network parameteres
        Uvec_pinn = [40]*4
        fdrop_pinn = 0.

        Uvec_ebm = [5]*3
        fdrop_ebm = 0.5
        
        return Uvec_pinn, Uvec_ebm, fdrop_pinn, fdrop_ebm
    
    def init_train_pars(self, bs_coll = 100, bs_train = 200, lr_pinn = 2e-3, lr_ebm = 2e-3, ld_fac = 1, lf_fac = 1,
                        Nebm = 2000, i_init_ebm = -1, Npinn = 3000):  #training parameters
        
        return bs_train, bs_coll, lr_pinn, lr_ebm, ld_fac, lf_fac, Nebm, i_init_ebm, Npinn
        
    def get_fx(self):  #definition of function
        print('not implemented!')
        
    def get_loss_d(self, n_opt = 'G0'):
        def loss_d(y_true, y_net):
            return (y_true.squeeze() - y_net.squeeze())**2
        return loss_d
        
    def get_loss_f(self):  #definition of pde loss
        print('not implemented!')
        
    def set_temp(self, t, x, dpar):
        return -1
        
    def get_meas_eq(self, y_opt=-1):
        def meas_eq(x, mpar=-1):
            if y_opt == -1:
                return x
        return meas_eq
 
        
 
        
class ds_exp_1d(pebm_dataset):
    def __init__(self, pars, dpar):
        if type(dpar) == str:
            dpar = [0.3]
        pebm_dataset.__init__(self, pars, dpar)
        #self.dpar = [0.3]
        
    def init_data_ranges(self):
        tmin, tmax = (0,), (10,)        
        tmin_coll, tmax_coll = (0,), (15,)    
        nfac = 1
        N_train = 200
        N_coll = 2000
        return tmin, tmax, tmin_coll, tmax_coll, nfac, N_train, N_coll
    
    def get_fx(self):
        return lambda t: torch.exp(self.dpar[0]*t)
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            lf = (xdot - dpar[0]*x)**2
            return lf
        return loss_f
    
    
    
    
    
class ds_sin_1d(pebm_dataset):
    def __init__(self, pars):
        pebm_dataset.__init__(self, pars)
        self.dpar = [0.5]
        
    def init_data_ranges(self):
        tmin, tmax = (0,), (20,)
        tmin_coll, tmax_coll = (0,), (25,)    
        nfac = 0.1
        
        N_train = 200
        N_coll = 2000
        
        return tmin, tmax, tmin_coll, tmax_coll, nfac, N_train, N_coll
        
    
    def get_fx(self):
        return lambda t: torch.sin(self.dpar[0]*t)
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]            
            xdot2 = torch.autograd.grad(xdot.sum(), t, create_graph=True)[0]
            lf = (xdot2 - (-dpar[0]**2)*x)**2
            
            return lf
        return loss_f

    
class ds_bessel_1d(pebm_dataset):
    def __init__(self, x_opt, dpar):
        pebm_dataset.__init__(self, x_opt, dpar) #TODO: fix inputs
        self.dpar = [0.7]
        
    def init_data_ranges(self):
        tmin, tmax = (0,), (10,)
        tmin_coll, tmax_coll = (0,), (15,)
        nfac = 0.03
        
        N_train = 200
        N_coll = 2000
        
        
        return tmin, tmax, tmin_coll, tmax_coll, nfac, N_train, N_coll
    
        
    def get_fx(self):
        def bessel(t):
            t = t.numpy()
            buf = sp.special.jv(1, self.dpar[0]*t)
            return torch.tensor(buf)
        return bessel
    
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            xdot2 = torch.autograd.grad(xdot.sum(), t, create_graph=True)[0]
            
            lf = (t**2*xdot2 + t*xdot + (dpar[0]**2*t**2 - 1)*x)**2
            return lf
        return loss_f
    
    
    
class ds_sin_2d(pebm_dataset):
    def __init__(self, pars):
        pebm_dataset.__init__(self, pars)
        self.dpar = [0.5, 0.3]
        
    def init_data_ranges(self):
        tmin, tmax = (0, 0), (20, 20)
        tmin_coll, tmax_coll = (0, 0), (30, 30)    
        nfac = 0.3
        
        N_train = 32**2
        N_coll = 25**2
        
        return tmin, tmax, tmin_coll, tmax_coll, nfac, N_train, N_coll
        
    
    def get_fx(self):
        return lambda t: torch.sin(self.dpar[0]*t[:,0])*torch.sin(self.dpar[1]*t[:,1])
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            x = x.squeeze()
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            xdot0 = xdot[:,0]
            xdot1 = xdot[:,1]
            xdot0_2 = torch.autograd.grad(xdot0.sum(), t, create_graph=True)[0][:,0]
            xdot1_2 = torch.autograd.grad(xdot1.sum(), t, create_graph=True)[0][:,1]
            lf0 = (xdot0_2 - (-dpar[0]**2)*x)**2
            lf1 = (xdot1_2 - (-dpar[1]**2)*x)**2            
            lf = lf0 + lf1            
            
            return lf
        return loss_f
    
class ds_sin_exp_2d(pebm_dataset):
    def __init__(self, pars):
        pebm_dataset.__init__(self, pars)
        self.dpar = [0.3, 0.5]
        
    def init_data_ranges(self):
        tmin, tmax = (0, 0), (10, 20)
        tmin_coll, tmax_coll = (0, 0), (15, 25)     
        nfac = 1
        
        N_train = 32**2
        #N_coll = 45**2
        N_coll = 100**2
        
        return tmin, tmax, tmin_coll, tmax_coll, nfac, N_train, N_coll
            
    
    def get_fx(self):
        return lambda t: torch.exp(self.dpar[0]*t[:,0])*torch.sin(self.dpar[1]*t[:,1])
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            x = x.squeeze()
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            xdot0 = xdot[:,0]
            xdot1 = xdot[:,1]
            xdot0_2 = torch.autograd.grad(xdot0.sum(), t, create_graph=True)[0][:,0]
            xdot1_2 = torch.autograd.grad(xdot1.sum(), t, create_graph=True)[0][:,1]
            lf0 = (xdot0 - dpar[0]*x)**2
            lf1 = (xdot1_2 - (-dpar[1]**2)*x)**2
            lf = lf0 + lf1
            
            return lf
        return loss_f
    
    
class ds_exp_2d(pebm_dataset):
    def __init__(self, pars):
        pebm_dataset.__init__(self, pars)
        self.dpar = [0.3, 0.2]
        
    def init_data_ranges(self):
        tmin, tmax = (0, 0), (10, 10)
        tmin_coll, tmax_coll = (0, 0), (15, 15)     
        nfac = 0.5
        
        N_train = 32**2
        N_coll = 45**2
        
        return tmin, tmax,tmin_coll, tmax_coll, nfac, N_train, N_coll
    
    def get_fx(self):
        return lambda t: torch.exp(self.dpar[0]*t[:,0])*torch.sin(self.dpar[1]*t[:,1])
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            x = x.squeeze()
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            xdot0 = xdot[:,0]
            xdot1 = xdot[:,1]
            lf0 = (xdot0 - dpar[0]*x)**2
            lf1 = (xdot1 - dpar[1]*x)**2
            lf = lf0 + lf1
            
            return lf
        return loss_f
    
############################################
############################################
############################################
 
    
    
    
class ds_navier_stokes(pebm_dataset):
    def __init__(self, pars):
        pebm_dataset.__init__(self, pars,-1)
        self.dpar = [1, 0.01]
        

    def init_network_pars(self):  #network parameteres
        Uvec_pinn = [20]*8
        fdrop_pinn = 0.

        Uvec_ebm = [5]*3
        fdrop_ebm = 0.5
        
        return Uvec_pinn, Uvec_ebm, fdrop_pinn, fdrop_ebm
    
    def init_data_ranges(self):
        tmin, tmax = (0, 0, 0), (1, 1, 1)
        tmin_coll, tmax_coll = (0, 0, 0), (1, 1, 1)      
        nfac = 0.05
        N_train = 5000
        N_coll = 10**2
        
        return tmin, tmax, tmin_coll, tmax_coll, nfac, N_train, N_coll

    
    def set_temp(self, t, x, dpar):
        self.u, self.v, self.p, self.f, self.g = get_aux(t, x, dpar[0], dpar[1])
    
    def get_meas_eq(self, yopt=-1):
        def meas_eq(x, mpar=-1):
            N = x.shape[0]
            return torch.stack((self.u[-N:],self.v[-N:])).T
        return meas_eq

    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
                  
            loss_f = self.f**2
            loss_g = self.g**2
            
            return loss_f + loss_g
        return loss_f

# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy as sp

from .utils import set_par
from .utils_navier_stokes import get_aux

def get_ds(pars):
    x_opt = pars['x_opt']
    if x_opt == 1:
        ds = ds_exp_1d(pars)
    if x_opt == 2:
        ds = ds_sin_1d(pars)
    if x_opt == 3:
        ds = ds_bessel_1d(pars)
    if x_opt == 4:
        ds = ds_sin_2d(pars)
    if x_opt == 5:
        ds = ds_sin_exp_2d(pars)
    if x_opt == 6:
        ds = ds_exp_2d(pars)    
    if x_opt == 101:
        ds = ds_navier_stokes(pars)
        
    return ds



class pebm_dataset():
    def __init__(self, pars):
        self.x_opt = pars['x_opt']
        self.dpar = pars['dpar']
        
        self.init_train_pars(pars)
        self.init_data_ranges(pars)
        self.init_network_pars(pars)
        
        pars['nfac'] = pars['nfac0']*pars['fnoise']
        
    def init_data_ranges(self, pars):  #ranges of input data and normalizing const.
        print('not implemented!')
    
    def init_network_pars(self, pars):  #network parameteres        
        set_par(pars, 'Uvec_pinn',  [40]*4)
        set_par(pars, 'fdrop_pinn', 0.)
        set_par(pars, 'Uvec_ebm', [5]*3)
        set_par(pars, 'fdrop_ebm', 0.5)
    
    def init_train_pars(self, pars):  #training parameters
    
        set_par(pars, 'bs_coll', 100)
        set_par(pars, 'bs_train', 200)
        set_par(pars, 'lr_pinn', 2e-3)
        set_par(pars, 'lr_ebm', 2e-3)
        set_par(pars, 'ld_fac', 1)
        set_par(pars, 'lf_fac', 1)
        set_par(pars, 'Nebm', 2000)
        set_par(pars, 'i_init_ebm', -1)
        set_par(pars, 'Npinn', 3000)
        
        
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
    def __init__(self, pars):
        set_par(pars, 'dpar', [0.3])
        pebm_dataset.__init__(self, pars)
        
    def init_data_ranges(self, pars):
        
        set_par(pars, 'tmin', (0,))        
        set_par(pars, 'tmax', (10,))        
        set_par(pars, 'tmin_coll', (0,))      
        set_par(pars, 'tmax_coll', (15,))
        set_par(pars, 'nfac0', 1)
        set_par(pars, 'N_train', 200)        
        set_par(pars, 'N_coll', 2000)  
        
    
    def get_fx(self):
        return lambda t: torch.exp(self.dpar[0]*t)
    
    def get_loss_f(self):
        def loss_f(t, x, dpar):
            xdot = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
            lf = (xdot - dpar[0]*x)**2
            return lf
        return loss_f
    
    
    


    
class ds_bessel_1d(pebm_dataset):
    def __init__(self, pars):
        set_par(pars, 'dpar', [0.7])
        pebm_dataset.__init__(self, pars)
        
    def init_data_ranges(self, pars):
        set_par(pars, 'tmin', (0,))
        set_par(pars, 'tmax', (10,))        
        set_par(pars, 'tmin_coll', (0,))
        set_par(pars, 'tmax_coll', (15,))
        set_par(pars, 'nfac0', 0.03)
        set_par(pars, 'N_train', 200)        
        set_par(pars, 'N_coll', 2000)
    
        
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
    
    
    

    
############################################
############################################
############################################
 
    
    
    
class ds_navier_stokes(pebm_dataset):
    def __init__(self, pars):        
        set_par(pars, 'dpar', [1, 0.01])
        pebm_dataset.__init__(self, pars)
        

    def init_network_pars(self, pars):  #network parameters
        set_par(pars, 'Uvec_pinn',  [20]*8)
        set_par(pars, 'fdrop_pinn', 0.)
        set_par(pars, 'Uvec_ebm', [5]*3)
        set_par(pars, 'fdrop_ebm', 0.5)
        
    
    def init_data_ranges(self, pars):        
        set_par(pars, 'tmin', (0, 0, 0))
        set_par(pars, 'tmax', (1, 1, 1))        
        set_par(pars, 'tmin_coll', (0, 0, 0))
        set_par(pars, 'tmax_coll', (1, 1, 1))
        set_par(pars, 'nfac0', 0.05)
        set_par(pars, 'N_train', 5000)        
        set_par(pars, 'N_coll', 10**2)
        
    
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

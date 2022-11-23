# -*- coding: utf-8 -*-

import numpy as np
import torch
import scipy
import scipy.io
import torch.nn as nn

#for details on the Navier-Stokes dataset see https://github.com/maziarraissi/PINNs/
#############
#############
# MIT License

# Copyright (c) 2018 maziarraissi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#############
#############

def load_ns_data(N_train = 4000, N_test = 1000):
    # from Github: load data
    # Load Data
    data = scipy.io.loadmat('../data/cylinder_nektar_wake.mat')
           
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    T_train = 160
    train_in, train_out = ns_rearrange_data(XX[:,:T_train], YY[:,:T_train], TT[:,:T_train], UU[:,:T_train], VV[:,:T_train], PP[:,:T_train], N_train, T_train)
    test_in, test_out = ns_rearrange_data(XX[:,T_train:], YY[:,T_train:], TT[:,T_train:], UU[:,T_train:], VV[:,T_train:], PP[:,T_train:], N_test, T-T_train)
    
    
    return train_in, train_out, test_in, test_out

def ns_rearrange_data(XX, YY, TT, UU, VV, PP, N, T):
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1
    
    # Training Data    
    idx = np.random.choice(N*T, N, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    
    d_in = np.concatenate([x_train, y_train, t_train], 1)
    d_out = np.concatenate([u_train, v_train], 1)
    
    return torch.tensor(d_in).float(), torch.tensor(d_out).float()
    

#get pde loss
def get_aux(inputs, outputs_nn, l1, l2):
    #l1 = 1
    #l2 = 0.01
    psi = outputs_nn[:,0]
    p = outputs_nn[:,1]
    
    u = torch.autograd.grad(psi.sum(), inputs, create_graph=True)[0][:,1]
    v = -torch.autograd.grad(psi.sum(), inputs, create_graph=True)[0][:,0]
    
    p_xy = torch.autograd.grad(p.sum(), inputs, create_graph=True)[0]
    p_x = p_xy[:,0]
    p_y = p_xy[:,1]
    
    u_xyt = torch.autograd.grad(u.sum(), inputs, create_graph=True)[0]  #x,y,t 
    u_x = u_xyt[:,0]
    u_y = u_xyt[:,1]
    u_t = u_xyt[:,2]
    u_xx = torch.autograd.grad(u_x.sum(), inputs, create_graph = True)[0][:,0]
    u_yy = torch.autograd.grad(u_y.sum(), inputs, create_graph = True)[0][:,1]
    
    v_xyt = torch.autograd.grad(v.sum(), inputs, create_graph=True)[0]
    v_x = v_xyt[:,0]
    v_y = v_xyt[:,1]
    v_t = v_xyt[:,2]
    v_xx = torch.autograd.grad(v_x.sum(), inputs, create_graph = True)[0][:,0]
    v_yy = torch.autograd.grad(v_y.sum(), inputs, create_graph = True)[0][:,1]
    
    f = u_t + l1*(u*u_x + v*u_y) + p_x - l2*(u_xx + u_yy)
    g = v_t + l1*(u*v_x + v*v_y) + p_y - l2*(v_xx + v_yy)
    
    return u, v, p, f, g


class Net_NS(nn.Module):
    def __init__(self, device):
        super(Net_NS, self).__init__()      
        Nq = 30#20
        
        self.dpar = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.mpar = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        self.opar = nn.Parameter(torch.tensor([0.0]))
        self.Wlist = []
        self.blist = []
        
        #xavier initialization of weights
        winit = torch.sqrt(torch.tensor(1/Nq))
        winit0 = torch.sqrt(torch.tensor(2/(3+Nq)))
        winitl = torch.sqrt(torch.tensor(2/(Nq+2)))
        
        self.Wlist.append(nn.Parameter(winit0*torch.randn(3,Nq)))
        self.blist.append(nn.Parameter(torch.zeros(Nq)))
        for j in range(3):#7):
            self.Wlist.append(nn.Parameter(winit*torch.randn(Nq,Nq)))            
            self.blist.append(nn.Parameter(torch.zeros(Nq)))
        self.Wlist.append(nn.Parameter(winitl*torch.randn(Nq,2)))        
        self.blist.append(nn.Parameter(torch.zeros(2)))
        
        self.register_parameters()
        
        #min/max values for normalization of inputs
        self.lb = torch.tensor([1., -2., 0.]).to(device)
        self.ub = torch.tensor([8.0, 2.0, 19.9]).to(device)
        
    def register_parameters(self):
        for j in range(len(self.Wlist)):
            self.register_parameter('W'+str(j), self.Wlist[j])
            self.register_parameter('b'+str(j), self.blist[j])
            

    def forward(self, X):
        act = torch.tanh #activation function
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize input
        
        for j in range(4):#8):
            X = act(X.mm(self.Wlist[j]) + self.blist[j])        
        X = X.mm(self.Wlist[j+1]) + self.blist[j+1]
        return X
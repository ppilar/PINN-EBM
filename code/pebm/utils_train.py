# -*- coding: utf-8 -*-

import os
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .plots import plot_ebm, plot_pinn
from .utils_navier_stokes import Net_NS, load_ns_data
from .utils import get_label

#######
####### initialization


def get_ytrain_etc(f_train, f_test, noise, x_opt):
    #generate noisy measurements
    
    N_train = f_train.shape[0]
    N_test = f_test.shape[0]
    ydim = 1 if f_train.ndim == 1 else f_train.shape[1]
    
    noise_train = noise.sample(N_train*ydim).reshape((N_train,ydim)).squeeze()
    y_train = f_train + noise_train
    
    noise_test = noise.sample(N_test*ydim).reshape((N_test,ydim)).squeeze()
    y_test = f_test + noise_test
    
    ymin = torch.min(y_train)
    ymax = torch.max(y_train)
    
    return y_train, y_test, noise_train, noise_test, ymin, ymax


def init_nets(dim, tmin_coll, tmax_coll, ymin, ymax, Uvec_pinn, fdrop_pinn, Uvec_ebm, fdrop_ebm, x_opt, device):
    #initialize nets for PINN and EBM
    
    if x_opt == 101:        
        pinn_bounds = (tmin_coll, tmax_coll, ymin.item(), ymax.item())
        net_pinn_ebm = Net_NS(device).to(device)
        net_pinn0 = Net_NS(device).to(device)
        net_pinn_offset = Net_NS(device).to(device)
        net_pinn0_2 = Net_NS(device).to(device)
        net_nn = Net_NS(device).to(device)
    else:
        pinn_bounds = (tmin_coll, tmax_coll, ymin.item(), ymax.item())
        net_pinn_ebm = Net(Uvec_pinn, fdrop=fdrop_pinn, input_dim = dim, bounds = pinn_bounds, device = device)
        net_pinn0 = Net(Uvec_pinn, fdrop=fdrop_pinn, input_dim = dim, bounds = pinn_bounds, device = device)
        net_pinn_offset = Net(Uvec_pinn, fdrop=fdrop_pinn, input_dim = dim, bounds = pinn_bounds, device = device)
        net_pinn0_2 = Net(Uvec_pinn, fdrop=fdrop_pinn, input_dim = dim, bounds = pinn_bounds, device = device)
        net_nn = Net(Uvec_pinn, fdrop=fdrop_pinn, input_dim = dim, bounds = pinn_bounds, device = device)
    ebm_bounds = (-ymax.item(), ymax.item(), 0, 10)
    net_ebm = Net(Uvec_ebm, fdrop=fdrop_ebm, bounds = ebm_bounds).to(device)
    
    nets_pinn = [net_pinn0, net_pinn_ebm, net_pinn_offset, net_pinn0_2, net_nn]
    return nets_pinn, net_ebm 




def adjust_data(jN, dim, x_opt, t_coll, t_train, t_test, y_train, y_test, device):
    #put data on correct device and fix dimensions
    
    if jN == 0:
        t_coll = t_coll.to(device)
        t_train = t_train.to(device).float()
        t_test = t_test.to(device)
    y_train = y_train.to(device).float()
    y_train = y_train.unsqueeze(1)
    if x_opt != 101:
        y_test = y_test.to(device).unsqueeze(1).float()
    else:
        y_test = y_test.to(device).float()
    
    if dim == 1 and jN == 0:
        t_train = t_train.unsqueeze(1)
        if x_opt != 101:  t_test = t_test.unsqueeze(1)
        
    return t_coll, t_train, t_test, y_train, y_test 


def get_optimizers(nets_pinn, net_ebm, lr_pinn, lr_ebm):
    #initialize optimizers
    
    #Adam
    optimizer_pinn_ebm = optim.Adam(nets_pinn[1].parameters(), lr=lr_pinn) #1
    optimizer_pinn0 = optim.Adam(nets_pinn[0].parameters(), lr=lr_pinn) #0
    optimizer_pinn_offset = optim.Adam(nets_pinn[2].parameters(), lr=lr_pinn) #2
    optimizer_pinn0_2 = optim.Adam(nets_pinn[3].parameters(), lr=lr_pinn) #3
    optimizer_nn = optim.Adam(nets_pinn[4].parameters(), lr=lr_pinn) #4
    optimizer_ebm = -1# optim.Adam(net_ebm.parameters(), lr=lr_ebm)
    
    optimizers_pinn = [optimizer_pinn0, optimizer_pinn_ebm, optimizer_pinn_offset, optimizer_pinn0_2, optimizer_nn]
    
    return optimizers_pinn, optimizer_ebm



################
###############

def get_residuals(y_train_batch, y_net_train):
    #get residual between training data and PINN prediction
    
    res = (y_train_batch.squeeze() - y_net_train.squeeze())
    #res = (res/(0.3*(torch.abs(y_net)+1e-3).squeeze()))
    return res.flatten()

def get_train_batches(trainiter, trainloader, epoch_data):
    #load next training batch
    
    new_epoch = -1
    try:
        train_batch = next(trainiter)
    except StopIteration:
        trainiter = iter(trainloader)
        train_batch = next(trainiter)
        epoch_data += 1
        new_epoch = 1
    return train_batch[0], train_batch[1], epoch_data, new_epoch

def initialize_data(tmin, tmax, tmin_coll, tmax_coll, N_train, N_coll, x_opt, fx, fmeq0):
    #initialize training and test data
    
    if x_opt != 101:
        t_train, t_test, t_coll = get_ttrain_tcoll(tmin, tmax, tmin_coll, tmax_coll, N_train, N_coll)
        x_train = fx(t_train)
        f_train = fmeq0(x_train)    
        x_test = fx(t_test)
        f_test = fmeq0(x_test)
    else:
        t_train, f_train, t_test, f_test = load_ns_data()
        t_coll = t_train
        
    return t_train, f_train, t_test, f_test, t_coll

def get_ttrain_tcoll(tmin, tmax, tmin_coll, tmax_coll, N_train, N_coll, N_test = 49):
    #generate grids for training points and collocation points
    
    dim = len(tmin)

    if dim == 1:
        t_train = torch.linspace(tmin[0], tmax[0], N_train)
        t_test = torch.linspace(tmax[0], tmax_coll[0], N_test)
        coff = tmax_coll[0]*0.02
        t_coll = torch.linspace(tmin_coll[0]-coff, tmax_coll[0]+coff, N_coll)
    if dim == 2:
        sN_train = int(np.sqrt(N_train))
        sN_test = int(np.sqrt(N_test))
        sN_coll = int(np.sqrt(N_coll))
        
        t_train = torch.zeros(N_train, dim)
        t_train0 = torch.linspace(tmin[0], tmax[0], sN_train)
        t_train1 = torch.linspace(tmin[1], tmax[1], sN_train)    
        buf0, buf1 = torch.meshgrid(t_train0, t_train1)
        t_train[:,0], t_train[:,1] = buf0.flatten(), buf1.flatten()
        
        t_test = torch.zeros(N_test, dim)
        t_test0 = torch.linspace(tmax[0], tmax_coll[0], sN_test)
        t_test1 = torch.linspace(tmax[1], tmax_coll[1], sN_test    )
        buf0, buf1 = torch.meshgrid(t_test0, t_test1)
        t_test[:,0], t_test[:,1] = buf0.flatten(), buf1.flatten()        
        
        
        t_coll = torch.zeros(N_coll, dim)
        t_coll0 = torch.linspace(tmin_coll[0], tmax_coll[0], sN_coll)
        t_coll1 = torch.linspace(tmin_coll[0], tmax_coll[0], sN_coll)    
        buf0, buf1 = torch.meshgrid(t_coll0, t_coll1)
        t_coll[:,0], t_coll[:,1]  = buf0.flatten(), buf1.flatten()
    
    return t_train, t_test, t_coll



def get_fmeq_off(fmeq0, off):
    #adjust measurement equation to take offset parameter into account
    
    def meas_eq(x, mpar):
        return fmeq0(x, mpar) + off
    return meas_eq



def get_test_error(jm, ds, net_pinn, ebm, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf,  device):
    #calculate RMSE and logL on test data
    
    net_pinn.eval()
    ebm.net_ebm.eval()
    
    t_test.requires_grad = True
    x_net_test = net_pinn.forward(t_test)
    ds.set_temp(t_test, x_net_test, net_pinn.dpar)
    f_net_test = fmeq(x_net_test, net_pinn.mpar)
    res_test =  get_residuals(y_test, f_net_test).float()
    
    t_train.requires_grad = True
    x_net_train = net_pinn.forward(t_train)
    ds.set_temp(t_train, x_net_train, net_pinn.dpar)
    f_net_train = fmeq(x_net_train, net_pinn.mpar)
    res_train = get_residuals(y_train, f_net_train).float()
    
    if ds.x_opt != 101:
        rmse_train = torch.mean(torch.abs(x_net_train.detach().cpu().squeeze() - f_train))
        rmse_test = torch.mean(torch.abs(x_net_test.detach().cpu().squeeze() - f_test))
    else:        
        rmse_train = torch.mean(torch.abs((f_net_train  - net_pinn.opar).detach().cpu().squeeze() - f_train))
        rmse_test = torch.mean(torch.abs((f_net_test  - net_pinn.opar).detach().cpu().squeeze() - f_test))
    
  
    std_G = res_train.std()
    logL_G_train = -1/(2*std_G**2)*fld0(y_train, f_net_train).mean() - 0.5*torch.log(2*torch.pi*std_G**2)
    logL_G_test = -1/(2*std_G**2)*fld0(y_test, f_net_test).mean() - 0.5*torch.log(2*torch.pi*std_G**2)
    if jm in [1,3]:
        logL_ebm_train = -ebm.get_mean_NLL(res_train)
        logL_ebm_test = -ebm.get_mean_NLL(res_test)
    else:
        logL_ebm_train = torch.tensor(-1)
        logL_ebm_test = torch.tensor(-1)

    
    fl_train = torch.mean(flf(t_train, x_net_train, net_pinn.dpar))
    fl_test= torch.mean(flf(t_test, x_net_test, net_pinn.dpar))
    
    net_pinn.train()
    ebm.net_ebm.train()
    t_train.requires_grad = False
    
    return logL_G_train, logL_G_test, logL_ebm_train, logL_ebm_test, rmse_train, rmse_test, fl_train, fl_test



def get_ebm_curve(net_ebm, opar, noise, n_opt, n_fac, device):
    #collect ebm predictions over multiple runs
    
    y_plot = torch.linspace(-10,15,201, device=device)*n_fac
    pdf_true = noise.pdf(y_plot.cpu())
    
    net_ebm.eval()
    logits = net_ebm((y_plot - opar.item()).unsqueeze(1)).detach().cpu().squeeze()
    pdf_ebm = torch.exp(logits)
    return pdf_ebm, pdf_true, y_plot


def evaluate_rmse_dpde(net_pinn, ds, teval, flf, fx, device):
    net_pinn.zero_grad()
    net_pinn.eval()
    teval.requires_grad = True
    xeval = net_pinn.forward(teval)
    ds.set_temp(teval, xeval, net_pinn.dpar)
    fleval = flf(teval, xeval, net_pinn.dpar)        
    feval = fx(teval.detach().cpu()).to(device)
    rmse_eval = torch.sqrt((xeval-feval)**2)
    
    return np.array(fleval.detach().cpu()).squeeze(), np.array(rmse_eval.detach().cpu()).squeeze(), teval.detach().cpu()

#################
################# collect parameters etc.

def get_epoch_stats(losses, Nrun, itges, x_opt, jm, itest, iplot, ds, net_pinn, ebm, res, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, fx, fn, tmin, tmax, tmin_coll, tmax_coll, Npar, Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, fl_ges, device):
    #calculate some statistics after certain number of epochs
    
    loss_d0, loss_f0, loss_f1, loss = losses
    
    Jges[jm].append(loss.item())
    for jp in range(Npar):
        dparges[jm][jp].append(net_pinn.dpar[jp].detach().item())
    lossdges[jm].append(loss_d0.detach().item())
    lossfges[jm].append((loss_f0 + loss_f1).detach().item())
    
    #calculate test error
    if itges%(itest) == 0:# and x_opt != 101:
        logL_G_train, logL_G_test, logL_ebm_train, logL_ebm_test, rmse_train, rmse_test, fl_train, fl_test = get_test_error(jm, ds, net_pinn, ebm, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, device)
        logLG_ges[jm][0].append(logL_G_train.detach().item())
        logLG_ges[jm][1].append(logL_G_test.detach().item())
        logLebm_ges[jm][0].append(logL_ebm_train.detach().item())
        logLebm_ges[jm][1].append(logL_ebm_test.detach().item())
        rmse_ges[jm][0].append(rmse_train.detach().item())
        rmse_ges[jm][1].append(rmse_test.detach().item())
        fl_ges[jm][0].append(fl_train.detach().item())
        fl_ges[jm][1].append(fl_test.detach().item())
    
    if itges%(iplot) == 0 and itges > 0 and Nrun == 1:
        if jm in [1,3]:
            plot_ebm(ebm.net_ebm, res, net_pinn.opar, fn, res.min().item(), res.max().item(), '', device)
        plot_pinn((net_pinn,), fx, t_train, y_train, t_test, y_test, tmin, tmax, tmin_coll, tmax_coll, '', device)
        net_pinn.train().to(device)
        
    return Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, fl_ges


def calculate_stat(stat):
    s_est = stat[:,:,:,-1]
    mu = s_est.mean(2)
    std = s_est.std(2)
    return (s_est, mu, std)


def calculate_stats(dpargesges, logLG_gesges, logLebm_gesges, rmse_gesges, fleval_gesges, dpar_true):
    #calculate statistics over multiple runs  

    dpar_true = np.array(dpar_true)
    
    dpar_buf = np.zeros(dpargesges.shape)
    for j in range(dpar_true.size):
        dpar_buf[:,j,:,:] = np.abs(dpargesges[:,j,:,:] - dpar_true[j])
    tdpar = calculate_stat(dpar_buf)
    
    tlogLG = calculate_stat(logLG_gesges)
    tlogLebm = calculate_stat(logLebm_gesges)
    tlogL = tlogLG
    for j in range(3):
        tlogL[j][1] = tlogLebm[j][1]
        tlogL[j][3] = tlogLebm[j][3]
    
    trmse = calculate_stat(rmse_gesges)
    tfl = calculate_stat(fleval_gesges)

    return tdpar, tlogL, trmse, tfl


def print_stats(itges, j, jm, Jges, loss_d00, loss_f00, ld_fac, lf_fac, epoch_data, net_pinn):
    #print current values to console
    
    if len(Jges[jm]) > 50:
        ldm = np.mean(loss_d00[-50:])
        lfm = np.mean(loss_f00[-50:])
        ldstd = ld_fac*np.std(loss_d00[-50:])
        lfstd = lf_fac*np.std(loss_f00[-50:])
        loss_str = ''
    else:
        loss_str = ''
    
    
    print("\r", 'j' + str(j) 
          + ' itges' + str(itges)
          + ' epd' + str(epoch_data)
          + ' dpar: ' + str(round(net_pinn.dpar[0].item(),4))
          + ' dpar2: ' + str(round(net_pinn.dpar[1].item(),4))
          + ' opar: ' + str(round(net_pinn.opar.item(),3))
          + loss_str
          , end="")

def save_stats_to_txt(res, fname, input_path):
    tdpar, tlogL, trmse, tfl = calculate_stats(res.dpargesges, res.logLG_gesges, res.logLebm_gesges, res.rmse_gesges, res.fl_gesges, res.dpar)
    fname = 'x'+str(res.x_opt)+'n'+res.n_opt+'r'+str(res.Nrun)
    print_table(tdpar, trmse, tlogL, tfl, fname, input_path)


def print_table(tdpar, trmse, tlogL, tfl, fname, fpath):
    #create latex table entries of statistics 
    
    dig = 2
    fac = 1
    Npar = tdpar[0].shape[1]
    
    jm_vec = [3, 2, 0]
    model_str = '& & '
    for jm in jm_vec:
        model_str += get_label(jm) + ' & '
    
    jleg = ['$|\Delta \\lambda|$ ', 'RMSE', 'logL', '$|\Delta \\text{PDE}|$']
    jtt_ind = [1, 0, 1, 0]
    tvec = (tdpar, trmse, tlogL, tfl)
    facvec = (1e-2, 1, 1, 1e-2)
    with open(fpath+fname+'.txt', 'w') as f:
        original_stdout = sys.stdout 
        sys.stdout = f # Change the standard output to the file we created.
        
        print(model_str) #print model names
        
        for j in range(4):        
            
            fac = facvec[j]
            tbuf = tvec[j]
            jtt_vec = [jtt_ind[j]] if j > 0 else range(Npar) #choose between train and validation data
            for jtt in jtt_vec: #loop over multiple pars       
                print('&', jleg[j], end='')
                for jm in jm_vec: #loop over models
                    if jm == 3:
                        print(' & \\textbf{' + str(np.round(tbuf[1][jm,jtt]/fac, dig).item()) + '}$\pm$' + str(np.round(tbuf[2][jm,jtt]/fac, dig).item()), end='')
                    else:
                        print(' & ' + str(np.round(tbuf[1][jm,jtt]/fac, dig).item()) + '$\pm$' + str(np.round(tbuf[2][jm,jtt]/fac, dig).item()), end='')
                       
                print(' & ', '(e'+str(int(np.log10(fac)))+') \\\\')
        
        sys.stdout = original_stdout # Reset the standard output to its original value

#################
################# neural network definition

class Net(nn.Module): #map from t to x
    def __init__(self, Uvec = [], Npar=3, fdrop = 0, input_dim = 1, bounds = (0,0,0,0), device = 'cpu'):
        super(Net, self).__init__()
        if Uvec == []:
            U = 40
            self.Uvec = [U]*5
        else:
            self.Uvec = Uvec        
        self.dpar = nn.Parameter(1*torch.ones(Npar))
        self.mpar = nn.Parameter(0.1*torch.ones(Npar))
        self.opar = nn.Parameter(0*torch.ones(1))
        
           
        self.lb = torch.tensor(bounds[0]).float().to(device)
        self.ub = torch.tensor(bounds[1]).float().to(device)
        
        self.ylb = torch.tensor(bounds[2]).float().to(device)
        self.yub = torch.tensor(bounds[3]).float().to(device)
        
        output_dim = 1
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for j, hdim in enumerate(self.Uvec):
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))
        
        self.dr = nn.Dropout(fdrop)
     
    def forward(self, X):
        #normalize
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalize input   
        
        act = torch.tanh
        for j, layer in enumerate(self.layers[:-1]):
            X = act(layer(X))
            
        X = self.dr(X)  
        X = self.layers[-1](X)
        
        X = 0.5*((X + 1.0)*(self.yub - self.ylb)) + self.ylb #reverse normalization of output

        return X

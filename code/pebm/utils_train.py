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
from .noise import get_noise

#######
####### initialization


def get_ytrain_etc(f_train, f_test, noise, x_opt, prop_noise):
    #generate noisy measurements
    
    N_train = f_train.shape[0]
    N_test = f_test.shape[0]
    ydim = 1 if f_train.ndim == 1 else f_train.shape[1]
    
    noise_train = noise.sample(N_train*ydim).reshape((N_train,ydim)).squeeze()
    if prop_noise == 1:
        y_train = f_train + noise_train*f_train
    else:
        y_train = f_train + noise_train
    
    noise_test = noise.sample(N_test*ydim).reshape((N_test,ydim)).squeeze()
    if prop_noise == 1:
        y_test = f_test + noise_test*f_test
    else:
        y_test = f_test + noise_test
    
    ymin = torch.min(y_train)
    ymax = torch.max(y_train)
    
    return y_train, y_test, noise_train, noise_test, ymin, ymax


def init_nets(dim, pars, ymin, ymax, device):# tmin_coll, tmax_coll, ymin, ymax, Uvec_pinn, fdrop_pinn, Uvec_ebm, fdrop_ebm, x_opt, device):
    #initialize nets for PINN and EBM
    
    if pars['x_opt'] == 101:        
        pinn_bounds = (pars['tmin_coll'], pars['tmax_coll'], ymin.item(), ymax.item())
              
        net_pinn0 = Net_NS(device).to(device)
        net_pinn0_2 = Net_NS(device).to(device)
        net_pinn_offset = Net_NS(device).to(device)
        net_pinn_ebm = Net_NS(device).to(device)
        net_nn = Net_NS(device).to(device)
    else:
        pinn_bounds = (pars['tmin_coll'], pars['tmax_coll'], ymin.item(), ymax.item())
        
        net_pinn_ebm = Net(pars['Uvec_pinn'], fdrop=pars['fdrop_pinn'], input_dim = dim, bounds = pinn_bounds, device = device)
        net_pinn0 =    Net(pars['Uvec_pinn'], fdrop=pars['fdrop_pinn'], input_dim = dim, bounds = pinn_bounds, device = device)
        net_pinn_offset = Net(pars['Uvec_pinn'], fdrop=pars['fdrop_pinn'], input_dim = dim, bounds = pinn_bounds, device = device)
        net_pinn0_2 =  Net(pars['Uvec_pinn'], fdrop=pars['fdrop_pinn'], input_dim = dim, bounds = pinn_bounds, device = device)
        net_nn =       Net(pars['Uvec_pinn'], fdrop=pars['fdrop_pinn'], input_dim = dim, bounds = pinn_bounds, device = device)
    ebm_bounds = (-ymax.item(), ymax.item(), 0, 10)
    net_ebm = Net(pars['Uvec_ebm'], fdrop=pars['fdrop_ebm'], bounds = ebm_bounds).to(device)
    
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

def calculate_residuals(t, y, net_pinn, ds, fmeq, prop_noise):
    t.requires_grad = True
    x_net= net_pinn.forward(t)
    ds.set_temp(t, x_net, net_pinn.dpar)
    f_net = fmeq(x_net, net_pinn.mpar)
    residuals =  get_residuals(y, f_net, prop_noise).float()
    return residuals, f_net, x_net

def get_residuals(y_train_batch, y_net_train, prop_noise):
    #get residual between training data and PINN prediction
    
    if prop_noise == 0:
        res = (y_train_batch.squeeze() - y_net_train.squeeze())
    else:
        res = (y_train_batch.squeeze() - y_net_train.squeeze())/(torch.abs(y_net_train.squeeze()) + 1e-9)
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


def initialize_functions(ds, pars):#x_opt, n_opt, n_fac):
    fx = ds.get_fx() if pars['x_opt'] != 101 else -1 #function x(t)
    fn = get_noise(pars['n_opt'], pars['nfac']) #function to generate noise
    fmeq0 = ds.get_meas_eq() #function defining measurement equation
    flf = ds.get_loss_f() #pde loss
    fld0 = ds.get_loss_d() #data loss
    return fx, fn, fmeq0, flf, fld0

def initialize_data(pars, fx, fmeq0):
    #initialize training and test data
    
    if pars['x_opt'] != 101:
        t_train, t_test, t_coll = get_ttrain_tcoll(pars['tmin'], pars['tmax'], pars['tmin_coll'], pars['tmax_coll'], pars['N_train'], pars['N_coll'])
        x_train = fx(t_train)
        f_train = fmeq0(x_train)    
        x_test = fx(t_test)
        f_test = fmeq0(x_test)
    else:
        t_train, f_train, t_test, f_test = load_ns_data()
        t_coll = t_train
        
    return t_train, f_train, t_test, f_test, t_coll

def get_ttrain_tcoll(tmin, tmax, tmin_coll, tmax_coll, N_train, N_coll, N_test = 49, test_mode = 0):
    #generate grids for training points and collocation points
    
    dim = len(tmin)

    if dim == 1:
        t_train = torch.linspace(tmin[0], tmax[0], N_train)
        if test_mode == 0:
            t_test = torch.linspace(tmax[0], tmax_coll[0], N_test)
        elif test_mode == 1:
            t_test = torch.linspace(tmin[0], tmax[0], N_test)
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
        
        if test_mode == 0:
            t_test = torch.zeros(N_test, dim)
            t_test0 = torch.linspace(tmax[0], tmax_coll[0], sN_test)
            t_test1 = torch.linspace(tmax[1], tmax_coll[1], sN_test)
            buf0, buf1 = torch.meshgrid(t_test0, t_test1)
            t_test[:,0], t_test[:,1] = buf0.flatten(), buf1.flatten()
        elif test_mode == 1:
            _test = torch.zeros(N_test, dim)
            t_test0 = torch.linspace(tmin[0], tmax[0], sN_test)
            t_test1 = torch.linspace(tmin[1], tmax[1], sN_test)
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

def get_batches(batch, dim, trainiter, trainloader, epoch_data):
    t_batch = batch[0]
    if dim == 1: t_batch = t_batch.unsqueeze(1)
    t_train_batch, y_train_batch, epoch_data, new_epoch = get_train_batches(trainiter, trainloader, epoch_data)
    return t_batch, t_train_batch, y_train_batch, epoch_data, new_epoch


def initialize_EBM(res, ds, net_pinn, ebm, fmeq, fn, t_train, y_train, ymin, ymax, plot_path_jN, device, ebm_ubound, jm, prop_noise):
    tebm = time.time()
    Ntry = 0
    while (True):
        net_pinn.eval()
        
        residuals_ges = calculate_residuals(t_train, y_train, net_pinn, ds, fmeq, prop_noise)[0].detach()
        t_train.requires_grad = False
        net_pinn.train()
        
        
        res.tebm_avg, res.Jebm = ebm.initialize(residuals_ges, res.Jebm, res.tebm_avg)
        indicator0 = plot_ebm(ebm.net_ebm, residuals_ges, net_pinn.opar, fn, ymin, ymax, plot_path_jN + 'init', device)
        if ebm.indicator < ebm.thr:
            break
        Ntry += 1
        if Ntry == 3: #raising ebm_ubound can help EBM training to converge
            ebm_ubound = 5
            ebm.Nebm *= 2
            print('updated!')
    res.tebm_ges[jm] = time.time() - tebm
    
def calculate_losses(ds, net_pinn, ebm, fld0, flf, fmeq, t_batch, t_train_batch, y_train_batch, jm, itges, pars):
    
    t_batch.requires_grad = True
    t_train_batch.requires_grad = True

    #loss_f
    t_ges = torch.cat((t_batch, t_train_batch),0)
    x_ges = net_pinn.forward(t_ges)
    ds.set_temp(t_ges, x_ges, net_pinn.dpar)
    fl_ges = flf(t_ges, x_ges, net_pinn.dpar)
    f_net_train = fmeq(x_ges[t_batch.shape[0]:], net_pinn.mpar)
    residuals = get_residuals(y_train_batch, f_net_train, pars['prop_noise'])  
    
    loss_f0 = fl_ges[:t_batch.shape[0]].mean()
    loss_f1 = fl_ges[t_batch.shape[0]:].mean()
    
    
    #loss_d
    #calculate data loss
    if itges >= pars['i_init_ebm'] and jm in [1,3]:
        loss_d0 = ebm.get_mean_NLL(residuals)
    else:
        loss_d0 = fld0(y_train_batch, f_net_train).mean()
        
    
    #set weighting factor for pde loss
    if (jm == 1 or jm ==3):
        pars['lf_fac_l'] = pars['lf_fac2'] if itges >= pars['i_init_ebm'] else pars['lf_fac']
    else:
        pars['lf_fac_l'] = pars['lf_fac2'] if pars['lf_fac2_alt'] == -1 else pars['lf_fac2_alt']
            
    
    ### total loss
    loss = pars['ld_fac']*loss_d0 + pars['lf_fac_l']*(loss_f0) if jm != 4 else loss_d0

    return loss, loss_f0, loss_f1, loss_d0, loss, pars['lf_fac_l'], residuals
    
    


def get_test_error(jm, ds, net_pinn, ebm, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, prop_noise, device):
    #calculate RMSE and logL on test data
    
    net_pinn.eval()
    ebm.net_ebm.eval()
    
    
    res_test, f_net_test, x_net_test = calculate_residuals(t_test, y_test, net_pinn, ds, fmeq, prop_noise)
    res_train, f_net_train, x_net_train = calculate_residuals(t_train, y_train, net_pinn, ds, fmeq, prop_noise)     
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

def get_epoch_stats(losses, pars, itges, jm, ds, net_pinn, ebm, res, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, fx, fn, Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, fl_ges, device):
    #calculate some statistics after certain number of epochs
    
    loss_d0, loss_f0, loss_f1, loss = losses
    
    Jges[jm].append(loss.item())
    for jp in range(pars['Npar']):
        dparges[jm][jp].append(net_pinn.dpar[jp].detach().item())
    lossdges[jm].append(loss_d0.detach().item())
    lossfges[jm].append((loss_f0 + loss_f1).detach().item())
    
    #calculate test error
    if itges%(pars['itest']) == 0:# and x_opt != 101:
        logL_G_train, logL_G_test, logL_ebm_train, logL_ebm_test, rmse_train, rmse_test, fl_train, fl_test = get_test_error(jm, ds, net_pinn, ebm, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, pars['prop_noise'], device)
        logLG_ges[jm][0].append(logL_G_train.detach().item())
        logLG_ges[jm][1].append(logL_G_test.detach().item())
        logLebm_ges[jm][0].append(logL_ebm_train.detach().item())
        logLebm_ges[jm][1].append(logL_ebm_test.detach().item())
        rmse_ges[jm][0].append(rmse_train.detach().item())
        rmse_ges[jm][1].append(rmse_test.detach().item())
        fl_ges[jm][0].append(fl_train.detach().item())
        fl_ges[jm][1].append(fl_test.detach().item())
    
    if itges%(pars['iplot']) == 0 and itges > 0 and pars['Nrun'] == 1:
        if jm in [1,3]:
            plot_ebm(ebm.net_ebm, res, net_pinn.opar, fn, res.min().item(), res.max().item(), '', device)
        plot_pinn((net_pinn,), fx, t_train, y_train, t_test, y_test, pars['tmin'], pars['tmax'], pars['tmin_coll'], pars['tmax_coll'], '', device)
        net_pinn.train().to(device)
        
    return Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, fl_ges


def G_pdf(x, mu, std):
    return 1/np.sqrt(2*np.pi*std**2)*np.exp(-1/(2*std**2)*(x-mu)**2)

def store_rpdfs(res, ds, net_pinn, ebm, fmeq, t_test, t_train, y_test, y_train, device, jm, prop_noise):
    res_test = calculate_residuals(t_test, y_test, net_pinn, ds, fmeq, prop_noise)[0].detach().cpu() + net_pinn.opar.detach().cpu()     
    res_train = calculate_residuals(t_train, y_train, net_pinn, ds, fmeq, prop_noise)[0].detach().cpu() + net_pinn.opar.detach().cpu()
    
    if jm in [0,2]:
        res.rpdf = G_pdf(res.rvec, net_pinn.opar.detach().cpu().numpy(), res_train.std())
    elif jm in [1,3]:
        res.rpdf = ebm.get_pdf(torch.tensor(res.rvec, device=device).unsqueeze(1).float()).detach().cpu().numpy()
        
    
    res.residuals_train = res_train
    res.residuals_test = res_test
    
    t_train.requires_grad = False
    t_test.requires_grad = False  


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
          + ' lf_fac: ' + str(round(lf_fac,4))
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
    #jtt_ind = [1, 0, 1, 0]
    jtt_ind = [0, 1, 1, 0]
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

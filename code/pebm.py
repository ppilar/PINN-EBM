# -*- coding: utf-8 -*-
import os
os.environ["OMP_NUM_THREADS"] = '1'

import sys
import time
import random
import collections
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

from pebm.noise import *
from pebm.plots import *
from pebm.utils_train import *
from pebm.datasets import get_ds
from pebm.results import *
from pebm.ebm import *


#%%
#######
#######

    
if not 'rand_init' in locals(): rand_init, s = init_random_seeds(s=0) #set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not 'input_path' in locals(): input_path = set_input_path('../results/', 'test') #choose standard folder as input folder, if not otherwise specified
plot_path = input_path + 'plots/'
check_dirs(input_path, plot_path, replace=False)
exec(open(input_path+'input.py').read()) #run input file


############################################
############################################
############################################

    #initialize functions, arrays, ... (outside of loop)


dim = len(tmin) if (type(tmin) is tuple) else 1 #dimension of input space
fx = ds.get_fx() if x_opt != 101 else -1 #function x(t)
fn = get_noise(n_opt, n_fac) #function to generate noise
fmeq0 = ds.get_meas_eq() #function defining measurement equation
flf = ds.get_loss_f() #pde loss
fld0 = ds.get_loss_d() #data loss
t_train, f_train, t_test, f_test, t_coll  = initialize_data(tmin, tmax, tmin_coll, tmax_coll, N_train, N_coll, x_opt, fx, fmeq0) #initialize training and test data
ydim = 1 if f_train.ndim == 1 else f_train.shape[-1] #dimension of measurements
N_test = t_test.shape[0] #number of test points


res = Results(x_opt, n_opt, Npinn, Nrun, jmodel_vec, ds.dpar, itest, N_train, lf_fac2, ebm_ubound, Nmodel=5)

#%%
for jN in range(Nrun):  #loop over different runs
    print('run #'+str(jN))
    
#%%    
############################################
############################################
############################################


    #initialize noise, networks, ...
    plot_path_jN = plot_path + str(jN)
    res.init_run_results()


    
    y_train, y_test, noise, noise_test, ymin, ymax = get_ytrain_etc(f_train, f_test, fn, x_opt) #get actual data with noise
    if Nrun == 1 and x_opt != 101:  plot_data(dim, t_train, y_train, t_test, y_test, f_train) #plot data
    nets_pinn, _ = init_nets(dim, tmin_coll, tmax_coll, ymin, ymax, Uvec_pinn, fdrop_pinn, Uvec_ebm, fdrop_ebm, x_opt, device) # networks
    #ebm = EBM(ebm_ubound, net_ebm, Uvec_ebm, fdrop_ebm, lr_ebm,  batch_size_train, Nebm, device)
    t_coll, t_train, t_test, y_train, y_test = adjust_data(jN, dim, x_opt, t_coll, t_train, t_test, y_train, y_test, device) # move to device and adjust dimensions
    optimizers_pinn, _ = get_optimizers(nets_pinn, -1, lr_pinn, -1) #optimizers
    # trainloaders
    trainloader_coll = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        t_coll), batch_size=batch_size_coll, num_workers=0, shuffle=True)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        t_train, y_train), batch_size=batch_size_train, num_workers=0,  shuffle=True)
    trainiter = iter(trainloader)
    

#%%
############################################
############################################
############################################

    ## train PINN and EBM jointly
    
    for jm in jmodel_vec: #loop over different models
        itges = 0 #total number of iterations
        epoch_data = 0 #number of epochs
        tm = time.time()
        
        ################################
        ################################
        ################################  initialize pars
        
        #select correct network and optimizer
        net_pinn = nets_pinn[jm].to(device)
        optimizer_pinn = optimizers_pinn[jm]
        scheduler_pinn = torch.optim.lr_scheduler.ExponentialLR(optimizer_pinn, gamma=0.3)
        ebm = EBM(ebm_ubound, Uvec_ebm, ymax, fdrop_ebm, lr_ebm,  batch_size_train, Nebm, device)
            
        fmeq = get_fmeq_off(fmeq0, net_pinn.opar) if jm in [1,2] else fmeq0

        
        loss_d00 = []
        loss_f00 = []
        loss_f0 = 0
        net_pinn.train()
        
        if jm == 1 or jm == 3:
            if i_init_ebm == -1 or (len(res.Jges) > i_init_ebm and i_init_ebm != -1):
                ebm.net_ebm.train()
        else:
            ebm.net_ebm.eval()
            
            
        ################################
        ################################
        ################################  start training
        tlists = [ [] for j in range(7)]
        
#%%
        for j in range(Npinn): #loop over iterations
            tpinn = time.time()
            
            for i, batch in enumerate(trainloader_coll):
                tit = time.time()
                
                t0 = time.time()
                #initialize EBM
                if itges == i_init_ebm and jm in [1,3]:                    
                    tebm = time.time()
                    Ntry = 0
                    while (True):
                        net_pinn.eval()                        
                        t_train.requires_grad = True
                        x_net_train = net_pinn.forward(t_train)
                        ds.set_temp(t_train, x_net_train, net_pinn.dpar) 
                        y_net_train = fmeq(x_net_train, net_pinn.mpar).detach()
                        t_train.requires_grad = False                        
                        net_pinn.train()
                        residuals_ges = get_residuals(y_train, y_net_train)                        
                        
                        
                        res.tebm_avg, res.Jebm = ebm.initialize(residuals_ges, res.Jebm, res.tebm_avg)
                        indicator0 = plot_ebm(ebm.net_ebm, residuals_ges, net_pinn.opar, fn, ymin, ymax, plot_path_jN + 'init', device)
                        #print('\nindicator:',str(ebm.indicator),'ind0:',str(indicator0))
                        if ebm.indicator < ebm.thr:
                            break
                        Ntry += 1
                        if Ntry == 3: #raising ebm_ubound can help EBM training to converge
                            ebm_ubound = 5
                    res.tebm_ges[jm] = time.time() - tebm
                    
                #get batches
                t_batch = batch[0]
                if dim == 1: t_batch = t_batch.unsqueeze(1)
                t_batch.requires_grad = True
                t_train_batch, y_train_batch, epoch_data, new_epoch = get_train_batches(trainiter, trainloader, epoch_data)
                t_train_batch.requires_grad = True
                tlists[0].append(time.time() - t0)
                
                ################################
                ################################
                ################################  calculate losses
                
                #forward through network
                t1 = time.time()
                t_ges = torch.cat((t_batch, t_train_batch),0)
                x_ges = net_pinn.forward(t_ges)
                tlists[1].append(time.time() - t1)
                
                #calculate pde losses
                t2 = time.time()
                ds.set_temp(t_ges, x_ges, net_pinn.dpar)
                f_ges = flf(t_ges, x_ges, net_pinn.dpar)
                y_net_train = fmeq(x_ges[t_batch.shape[0]:], net_pinn.mpar)
                residuals = get_residuals(y_train_batch, y_net_train)                
                loss_f0 = f_ges[:t_batch.shape[0]].mean()
                loss_f1 = f_ges[t_batch.shape[0]:].mean()
                tlists[2].append(time.time() - t2)
                
                #calculate data loss
                t3 = time.time()
                if itges >= i_init_ebm and jm in [1,3]:
                    loss_d0 = ebm.get_mean_NLL(residuals)
                else:
                    loss_d0 = fld0(y_train_batch, y_net_train).mean()
                tlists[3].append(time.time() - t3)
                    
                
                
                ################################
                ################################
                ################################  total loss + update
                #set weighting factor for pde loss
                if (jm == 1 or jm ==3):
                    lf_fac_l = lf_fac2 if itges >= i_init_ebm else lf_fac
                else:
                    lf_fac_l = lf_fac2 if not 'lf_fac2_alt' in locals() else lf_fac2_alt
                
                ### total loss
                if jm != 4:
                    loss = ld_fac*loss_d0 + lf_fac_l*(loss_f0)
                else:
                    loss = loss_d0
                
                ### update network parameters
                t4 = time.time()
                optimizer_pinn.zero_grad()
                ebm.optimizer_ebm.zero_grad()
                loss.backward()
                optimizer_pinn.step()
                ebm.optimizer_ebm.step()
                tlists[4].append(time.time() - t4)
                
                
                ################################
                ################################
                ################################  pars and plotting
                loss_d00.append(loss_d0.item())
                loss_f00.append((loss_f0 + loss_f1).item())
                
                t5 = time.time()
                losses = [torch.abs(loss_d0), loss_f0, loss_f1, loss]
                Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, fl_ges = get_epoch_stats(losses, Nrun, itges, x_opt, jm, itest, iplot, ds, net_pinn, ebm, residuals, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, fx, fn, tmin, tmax, tmin_coll, tmax_coll, res.Npar, res.Jges, res.dparges, res.lossdges, res.lossfges, res.logLG_ges, res.logLebm_ges, res.rmse_ges, res.fl_ges, device)
                tlists[5].append(time.time() - t5) 
                tlists[6].append(time.time() - tit)
                
                
                itges += 1                
                if itges == Npinn:
                    break
                
                if itges == i_sched:
                    scheduler_pinn.step()
                    if jm in [1,3] and ebm.init==1:
                        ebm.scheduler_ebm.step()
                
            
            print_stats(itges, j, jm, Jges, loss_d00, loss_f00, ld_fac, lf_fac_l, epoch_data, net_pinn)            
            res.tpinn_avg[jm,jN] = (res.tpinn_avg[jm,jN]*(j) + time.time() - tpinn)/(j+1)
            if itges == Npinn:
                break

            
#%%    
        ####
        #### save pars           
        
        teval = torch.linspace(tmin[0],tmax[0],res.Nteval).unsqueeze(1).to(device)
        if x_opt != 101:
            res.fleval, res.rmse_eval, res.teval = evaluate_rmse_dpde(net_pinn, ds, teval, flf, fx, device)  
            
        
        res.tm_ges[jm] = time.time() - tm
        res.store_run_results(jm, jN)
        
        
#%%    
        ####
        #### plotting
        plot_ebm(ebm.net_ebm, residuals, net_pinn.opar, fn, ymin, ymax, plot_path_jN + 'end', device)  
        if jm == 1:
            buf, ebm_curve_true, y_ebm_plot = get_ebm_curve(ebm.net_ebm, net_pinn.opar, fn, n_opt, n_fac, device)
            res.ebm_curve_ges.append(buf)
        plot_pinn(nets_pinn, fx, t_train, y_train, t_test, y_test, tmin, tmax, tmin_coll, tmax_coll, plot_path_jN, device)
        plot_dpar_loss(dparges[jm], ds.dpar, lossdges[jm], lossfges[jm], ld_fac, lf_fac, Jges[jm], plot_path_jN)
        if jm == 2:
            plot_logL(jm, logLG_ges, logLebm_ges, rmse_ges, plot_path_jN, itest)
        
        print('')
        print('tm'+str(jm)+':', str(round(res.tm_ges[jm],3)))
#%%
if 'y_ebm_plot' in locals():
    plot_ebm_comp(y_ebm_plot, res.ebm_curve_ges, ebm_curve_true, input_path)
        
#%%
####
#### save res

fname = 'x'+str(x_opt)+'n'+n_opt+'r'+str(Nrun)
filename = input_path + fname + '.dat'
with open(filename,'wb') as f:
    pickle.dump(res, f)

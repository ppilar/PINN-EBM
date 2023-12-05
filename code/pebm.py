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
from pebm.utils import set_par
from pebm.utils_train import *
from pebm.datasets import get_ds
from pebm.results import *
from pebm.ebm import *



#%%


#%%
#######
#######

    
if not 'rand_init' in locals(): rand_init, s = init_random_seeds(s=42) #set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not 'input_path' in locals(): input_path = set_input_path('../results/', 'test') #choose standard folder as input folder, if not otherwise specified
plot_path = input_path + 'plots/'
check_dirs(input_path, plot_path, replace=False)
exec(open(input_path+'input.py').read()) #run input file




############################################
############################################
############################################

    #initialize functions, arrays, ... (outside of loop)


###
ds = get_ds(pars)
dim = len(pars['tmin']) if (type(pars['tmin']) is tuple) else 1 #dimension of input space
fx, fn, fmeq0, flf, fld0 = initialize_functions(ds, pars)
t_train, f_train, t_test, f_test, t_coll  = initialize_data(pars, fx, fmeq0) #initialize training and test data
ydim = 1 if f_train.ndim == 1 else f_train.shape[-1] #dimension of measurements
N_test = t_test.shape[0] #number of test points

res = Results(pars, Nmodel=5)

#%%
for jN in range(pars['Nrun']):  #loop over different runs
    print('run #'+str(jN))
    
#%%    
############################################
############################################
############################################


    #initialize noise, networks, ...
    plot_path_jN = plot_path + str(jN)
    res.init_run_results()

    ###
    #model.initialize_run()    
    y_train, y_test, noise, noise_test, ymin, ymax = get_ytrain_etc(f_train, f_test, fn, pars['x_opt'], pars['prop_noise']) #get actual data with noise
    if pars['Nrun'] == 1 and pars['x_opt'] != 101:  plot_data(dim, t_train, y_train, t_test, y_test, f_train) #plot data
    t_coll, t_train, t_test, y_train, y_test = adjust_data(jN, dim, pars['x_opt'], t_coll, t_train, t_test, y_train, y_test, device) # move to device and adjust dimensions
    nets_pinn, _ = init_nets(dim, pars, ymin, ymax, device) # networks
    optimizers_pinn, _ = get_optimizers(nets_pinn, -1, pars['lr_pinn'], -1) #optimizers
    ###
    
    noise_strength = torch.abs(noise).mean()/torch.abs(f_train).mean()
    print('noise_strength: %f' % (noise_strength))
    res.noise_strength_ges.append(noise_strength)
    
    # trainloaders
    trainloader_coll = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(t_coll), batch_size=pars['bs_coll'], num_workers=0, shuffle=True)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(t_train, y_train), batch_size=pars['bs_train'], num_workers=0,  shuffle=True)
    trainiter = iter(trainloader)
    

#%%
############################################
############################################
############################################

    ## train PINN and EBM jointly
    
    for jm in pars['jmodel_vec']: #loop over different models
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
        ebm = EBM(pars, ymax, device)
            
        fmeq = get_fmeq_off(fmeq0, net_pinn.opar) if jm in [1,2] else fmeq0

        
        loss_d00 = []
        loss_f00 = []
        loss_f0 = 0
        net_pinn.train()
        
        if jm == 1 or jm == 3:
            if pars['i_init_ebm'] == -1 or (len(res.Jges) > pars['i_init_ebm'] and pars['i_init_ebm'] != -1):
                ebm.net_ebm.train()
        else:
            ebm.net_ebm.eval()
            
            
        ################################
        ################################
        ################################  start training
        tlists = [ [] for j in range(7)]
        
#%%
        for j in range(pars['Npinn']): #loop over iterations
            tpinn = time.time()
            
            for i, batch in enumerate(trainloader_coll):
                tit = time.time()
                
                #initialize EBM
                if itges == pars['i_init_ebm'] and jm in [1,3]:
                    initialize_EBM(res, ds, net_pinn, ebm, fmeq, fn, t_train, y_train, ymin, ymax, plot_path_jN, device, pars['ebm_ubound'], jm, pars['prop_noise'])
                                        
                #get batches
                t_batch, t_train_batch, y_train_batch, epoch_data, new_epoch = get_batches(batch, dim, trainiter, trainloader, epoch_data)
                
                
                #  calculate losses                
                loss, loss_f0, loss_f1, loss_d0, loss, lf_fac_l, residuals = calculate_losses(ds, net_pinn, ebm, fld0, flf, fmeq, t_batch, t_train_batch, y_train_batch, jm, itges, pars)   
                
                
                ### update network parameters
                optimizer_pinn.zero_grad()
                ebm.optimizer_ebm.zero_grad()
                loss.backward()
                optimizer_pinn.step()
                ebm.optimizer_ebm.step()
                
                
                ################################
                ################################
                ################################  pars and plotting
                loss_d00.append(loss_d0.item())
                loss_f00.append((loss_f0 + loss_f1).item())
                
                t5 = time.time()
                losses = [torch.abs(loss_d0), loss_f0, loss_f1, loss]
                Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, fl_ges = get_epoch_stats(losses, pars, itges, jm, ds, net_pinn, ebm, residuals, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, flf, fx, fn, res.Jges, res.dparges, res.lossdges, res.lossfges, res.logLG_ges, res.logLebm_ges, res.rmse_ges, res.fl_ges, device)
                tlists[5].append(time.time() - t5) 
                tlists[6].append(time.time() - tit)
                
                
                itges += 1                
                if itges == pars['Npinn']:
                    break
                
                if itges == pars['i_sched']:
                    scheduler_pinn.step()
                    if jm in [1,3] and ebm.init==1:
                        ebm.scheduler_ebm.step()
                
            
            print_stats(itges, j, jm, Jges, loss_d00, loss_f00, pars['ld_fac'], pars['lf_fac_l'], epoch_data, net_pinn)            
            res.tpinn_avg[jm,jN] = (res.tpinn_avg[jm,jN]*(j) + time.time() - tpinn)/(j+1)
            if itges == pars['Npinn']:
                break

            
#%%    
        ####
        #### save pars           
        
        
        ebm.net_ebm.eval()
        net_pinn.eval()
        net_pinn.to(device)
        store_rpdfs(res, ds, net_pinn, ebm, fmeq, t_test, t_train, y_test, y_train, device, jm, pars['prop_noise'])
        

        
        
        
        teval = torch.linspace(pars['tmin'][0],pars['tmax'][0],res.Nteval).unsqueeze(1).to(device)
        if pars['x_opt'] != 101:
            res.fleval, res.rmse_eval, res.teval = evaluate_rmse_dpde(net_pinn, ds, teval, flf, fx, device)  
            
        
        res.tm_ges[jm] = time.time() - tm
        res.store_run_results(jm, jN)
        
        
#%%    
        ####
        #### plotting
        plot_ebm(ebm.net_ebm, residuals, net_pinn.opar, fn, ymin, ymax, plot_path_jN + 'end', device)  
        if jm == 1:
            buf, ebm_curve_true, y_ebm_plot = get_ebm_curve(ebm.net_ebm, net_pinn.opar, fn, pars['n_opt'], pars['nfac'], device)
            res.ebm_curve_ges.append(buf)
        plot_pinn(nets_pinn, fx, t_train, y_train, t_test, y_test, pars['tmin'], pars['tmax'], pars['tmin_coll'], pars['tmax_coll'], plot_path_jN, device)
        plot_dpar_loss(dparges[jm], ds.dpar, lossdges[jm], lossfges[jm], pars['ld_fac'], pars['lf_fac'], Jges[jm], plot_path_jN)
        if jm == 2:
            plot_logL(jm, logLG_ges, logLebm_ges, rmse_ges, plot_path_jN, pars['itest'])
        
        print('')
        print('tm'+str(jm)+':', str(round(res.tm_ges[jm],3)))
        
        
        
#%%
if 'y_ebm_plot' in locals():
    plot_ebm_comp(y_ebm_plot, res.ebm_curve_ges, ebm_curve_true, input_path)
        
#%%
####
#### save res

fname = 'x'+str(pars['x_opt'])+'n'+ pars['n_opt'] +'r'+str(pars['Nrun'])
filename = input_path + fname + '.dat'
with open(filename,'wb') as f:
    pickle.dump(res, f)

#exec(open('plot_results.py').read()) 
            
#%%
#plot_residuals(res)
    
    



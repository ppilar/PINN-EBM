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


#%%
#######
#######
s = np.random.randint(42*10**4)
#s = 1
torch.manual_seed(s)
np.random.seed(s)
random.seed(s)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## define setting, data
if not 'input_path' in locals():
    path0 = '../results/'
    input_path = path0 + 'test/'
    print(input_path)
    check_dirs(path0, input_path)
    
exec(open(input_path+'input.py').read())


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

#statistics
Nmodel = 4
Npar = len(ds.dpar)
dpargesges = np.zeros([Nmodel, Npar, Nrun, Npinn])
Jgesges, lossdgesges, lossfgesges = np.zeros([3, Nmodel, Nrun, Npinn])

logLG_gesges, logLebm_gesges, rmse_gesges = np.zeros([3, Nmodel, 2, Nrun, Npinn//itest])
ebm_curve_ges = []

#%%
for jN in range(Nrun):  #loop over different runs
    print('run #'+str(jN))
    
#%%    
############################################
############################################
############################################


    #initialize noise, networks, ...
    input_path_jN = input_path + str(jN)

    #initialize eval vars
    tebm_avg, tzero_avg = [0, 0]
    tpinn_avg = np.zeros((Nmodel, Nrun))
    Jebm = []
    tm_ges = np.zeros([Nmodel])
    lossdges, lossfges, Jges = [([],[],[],[]) for j in range(3)]
    logLG_ges, logLebm_ges, rmse_ges = [(([],[]),([],[]),([],[]),([],[])) for j in range(3)]  
    dparges = tuple([[[] for j in range(Npar)] for i in range(Nmodel)])



    #get actual data with noise
    y_train, y_test, noise, noise_test, ymin, ymax = get_ytrain_etc(f_train, f_test, fn, x_opt)
    #plot data
    if Nrun == 1 and x_opt != 101:  plot_data(dim, t_train, y_train, t_test, y_test, f_train)
    # networks
    nets_pinn, net_ebm = init_nets(dim, tmin_coll, tmax_coll, ymin, ymax, Uvec_pinn, fdrop_pinn, Uvec_ebm, fdrop_ebm, x_opt, device)
    # move to device and adjust dimensions
    t_coll, t_train, t_test, y_train, y_test = adjust_data(jN, dim, x_opt, t_coll, t_train, t_test, y_train, y_test, device)
    #optimizers
    optimizers_pinn, optimizer_ebm = get_optimizers(nets_pinn, net_ebm, lr_pinn, lr_ebm)
    # trainloaders
    trainloader_coll = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        t_coll), batch_size=batch_size_coll, num_workers=0, shuffle=True)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        t_train, y_train), batch_size=batch_size_train, num_workers=0,  shuffle=True)
    trainiter = iter(trainloader)
    

#%%
    tebm_ges = [0]
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
        if jm == 1:
            scheduler_ebm = torch.optim.lr_scheduler.ExponentialLR(optimizer_ebm, gamma=0.3)
            
        fmeq = get_fmeq_off(fmeq0, net_pinn.opar) if jm in [1,2] else fmeq0

        
        loss_d00 = []
        loss_f00 = []
        loss_f0 = 0
        net_pinn.train()
        
        if jm == 1:
            if i_init_ebm == -1 or (len(Jges) > i_init_ebm and i_init_ebm != -1):
                net_ebm.train()
        else:
            net_ebm.eval()
            
            
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
                    res_max = torch.max(torch.abs(res.min()), torch.abs(res.max()))
                    ebm_bounds = (-res_max.item(), res_max.item(), 0, 1)
                    net_ebm = Net(Uvec_ebm, fdrop=fdrop_ebm, bounds = ebm_bounds).to(device)
                    optimizer_ebm = optim.Adam(net_ebm.parameters(), lr=lr_ebm)
                    scheduler_ebm = torch.optim.lr_scheduler.ExponentialLR(optimizer_ebm, gamma=0.3)
                    
                    net_ebm, res, tebm_avg, Jebm = train_ebm(ds, net_pinn, net_ebm, optimizer_ebm, t_train, y_train, batch_size_train, fmeq, Nebm, Jebm, tebm_avg, device)
                    plot_ebm(net_ebm, res, net_pinn.opar, fn, ymin, ymax, input_path_jN + 'init', device)  
                    
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
                res = get_res(y_train_batch, y_net_train)                
                loss_f0 = f_ges[:t_batch.shape[0]].mean()
                loss_f1 = f_ges[t_batch.shape[0]:].mean()
                tlists[2].append(time.time() - t2)
                
                #calculate data loss
                t3 = time.time()
                if itges >= i_init_ebm and jm in [1,3]:
                    loss_d0 = get_mean_NLL(net_ebm, res, device = device)
                elif jm == 4:
                    loss_d0 = get_mean_NLL_Gmix(net_pinn, res, device = device)
                else:
                    loss_d0 = fld0(y_train_batch, y_net_train).mean()
                tlists[3].append(time.time() - t3)
                    
                
                
                ################################
                ################################
                ################################  total loss + update
                if itges > i_lf_fac and x_opt == 101 and jm == 1:
                    lf_fac_l = lf_fac2
                else:
                    lf_fac_l = lf_fac
                
                
                ### total loss
                loss = ld_fac*loss_d0 + lf_fac_l*(1*loss_f0 + loss_f1) 
                    
                ### update network parameters
                t4 = time.time()
                optimizer_pinn.zero_grad()
                optimizer_ebm.zero_grad()
                loss.backward()
                optimizer_pinn.step()
                optimizer_ebm.step()
                tlists[4].append(time.time() - t4)
                
                
                ################################
                ################################
                ################################  pars and plotting
                loss_d00.append(loss_d0.item())
                loss_f00.append((loss_f0 + loss_f1).item())
                
                t5 = time.time()
                losses = [torch.abs(loss_d0), loss_f0, loss_f1, loss]
                Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges = get_epoch_stats(losses, Nrun, itges, x_opt, jm, itest, iplot, ds, net_pinn, net_ebm, res, t_train, f_train, y_train, t_test, f_test, y_test, fmeq, fld0, fx, fn, tmin, tmax, tmin_coll, tmax_coll, Npar, Jges, dparges, lossdges, lossfges, logLG_ges, logLebm_ges, rmse_ges, device)
                tlists[5].append(time.time() - t5) 
                tlists[6].append(time.time() - tit)
                
                
                itges += 1                
                if itges == Npinn:
                    break
                
                if itges == i_sched:
                    scheduler_pinn.step()
                    if jm in [1,3]:
                        scheduler_ebm.step()
                
            
            print_stats(itges, j, jm, Jges, loss_d00, loss_f00, ld_fac, lf_fac_l, epoch_data, net_pinn)            
            tpinn_avg[jm,jN] = (tpinn_avg[jm,jN]*(j) + time.time() - tpinn)/(j+1)
            if itges == Npinn:
                break

            
#%%    
        ####
        #### save pars    
        for jp in range(Npar):     
            dpargesges[jm,jp,jN,:] = np.array(dparges[jm][jp][:])                
        Jgesges[jm,jN,:] = np.array(Jges[jm])
        lossdgesges[jm,jN,:] = np.array(lossdges[jm])
        lossfgesges[jm,jN,:] = np.array(lossfges[jm])
        logLG_gesges[jm,0,jN,:] = np.array(logLG_ges[jm][0])
        logLG_gesges[jm,1,jN,:] = np.array(logLG_ges[jm][1])
        logLebm_gesges[jm,0,jN,:] = np.array(logLebm_ges[jm][0])
        logLebm_gesges[jm,1,jN,:] = np.array(logLebm_ges[jm][1])
        rmse_gesges[jm,0,jN,:] = np.array(rmse_ges[jm][0])
        rmse_gesges[jm,1,jN,:] = np.array(rmse_ges[jm][1])
        
        
#%%    
        ####
        #### plotting
        plot_ebm(net_ebm, res, net_pinn.opar, fn, ymin, ymax, input_path_jN + 'end', device)  
        if jm == 1:
            buf, ebm_curve_true, y_ebm_plot = get_ebm_curve(net_ebm, net_pinn.opar, fn, n_opt, n_fac, device)
            ebm_curve_ges.append(buf)
        plot_pinn(nets_pinn, fx, t_train, y_train, t_test, y_test, tmin, tmax, tmin_coll, tmax_coll, input_path_jN, device)
        plot_dpar_loss(dparges[jm], ds.dpar, lossdges[jm], lossfges[jm], ld_fac, lf_fac, Jges[jm], input_path_jN)
        if jm == 2:
            plot_logL(jm, logLG_ges, logLebm_ges, rmse_ges, input_path_jN, itest)
        
        print('')
        tm_ges[jm] = time.time() - tm
        print('tm'+str(jm)+':', str(round(tm_ges[jm],3)))
#%%

plot_ebm_comp(y_ebm_plot, ebm_curve_ges, ebm_curve_true, input_path)
#%%
#tmeans = np.array(tlists)[:,-1000:].mean(1)
#print('tinit', 'tnet', 'tpde', 'tdata', 'tstep', 'tmisc', 'tges')
#print(np.round(tmeans,5))

####
#### save and plot statistics
f = open(input_path + 'results.txt', "w")
f.write('dpar0:\n')
f.write(str(dpargesges[0,:,:,-1]))
f.write('\ndpar:\n')
f.write(str(dpargesges[1,:,:,-1]))
f.close()
torch.save({"dpar0": dpargesges[0], "dpar": dpargesges[1]}, input_path + 'tensors.pt')

plot_dpar_statistics(dpargesges, ds.dpar, Jgesges, lossdgesges, lossfgesges, Npar, input_path)
plot_error_statistics(logLG_gesges, logLebm_gesges, rmse_gesges, input_path, itest)
#%%
####
#### print latex table of stats

tdpar, tlogL, trmse = calculate_stats(dpargesges, logLG_gesges, logLebm_gesges, rmse_gesges, ds.dpar)
fname = 'x'+str(x_opt)+'n'+n_opt+'r'+str(Nrun)
print_table(tdpar, trmse, tlogL, fname, input_path)
        
#%%
####
#### save results
filename = input_path + fname + '.dat'
with open(filename,'wb') as outfile:
    pickle.dump([dpargesges, lr_pinn, lr_ebm, Npar, input_path, logLG_gesges, logLebm_gesges, rmse_gesges, Jgesges, lossdgesges, lossfgesges, input_path,
                 ds.dpar, x_opt, n_opt, i_init_ebm, i_sched, ld_fac, lf_fac, itest,
                 Uvec_pinn, Uvec_ebm, fdrop_pinn, fdrop_ebm],outfile)

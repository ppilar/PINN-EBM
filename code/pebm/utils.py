# -*- coding: utf-8 -*-
import os
import random
import shutil
import torch
import numpy as np
import pickle

#initialize random seeds
def init_random_seeds(s=False):
    if type(s) == bool:
        s = s = np.random.randint(42*10**4)
        
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    
    rand_init = 1
    return rand_init, s

#create string leading to folder; also create folder
def set_input_path(path0, folder0, replace=True):
    input_path = path0 + folder0 + '/'
    check_dirs(path0, input_path, replace=True)    
    print(input_path)
    return input_path

#check if input_path exists and create if it does not
def check_dirs(path0, input_path, replace=False):
    
    if not os.path.exists(input_path):
        os.mkdir(input_path)    
    if not os.path.exists(input_path + 'input.py') or replace == True:
        shutil.copyfile(path0 + 'input.py', input_path + 'input.py')
        
def set_par(pars, key, val):
    if not key in pars: pars[key] = val
    
def init_compatible_pars(res): #make compatible with data from earlier version    
    res.pars = {}
    if res.x_opt == 1:
        res.pars['dpar'] = [0.3]        
    if res.x_opt == 3:
        res.pars['dpar'] = [0.7]
    if res.x_opt == 101:
        res.pars['dpar'] = [1, 0.01]
        
    
#extract metrics of interest from stored run data
def gather_run_data(input_path0, fname0, Nrun, par_vec, par_label, var_list=['dpar_abs', 'dpde', 'rmse', 'NLL'], tt_list=[0,0,1,1]):
    Nmodel=5
    Npar=2
    
    var_out_ges = np.zeros([len(var_list), len(par_vec), Nmodel, Npar, Nrun])
    var_out = np.zeros([len(var_list), len(par_vec), Nmodel, Nrun])
    sname_ges = []
    
    fname = fname0 + str(Nrun)
    for j, par in enumerate(par_vec):
        input_path = input_path0 + par_label + str(par) + '/'
        filename = input_path + fname + '.dat'
        with open(filename,'rb') as f:
            res = pickle.load(f)
            if not hasattr(res, 'pars'):
                init_compatible_pars(res)
        
        for jv, vname in enumerate(var_list):
            jp = tt_list[jv]
            
            if vname == 'dpar_abs':
                var_out_ges[jv,j,:,:,:] = res.dpargesges[:,:,:,-1]
                Ndpar = res.dpargesges.shape[1]
                for k in range(Ndpar):
                    var_out_ges[jv,j,:,k,:] = np.abs(var_out_ges[jv,j,:,k,:] - res.pars['dpar'][k])
                sname_ges.append('$|\Delta \lambda|$')
            if vname == 'rmse':
                var_out_ges[jv,j,:,:,:] = res.rmse_gesges[:,:,:,-1]
                sname_ges.append('RMSE')
            if vname == 'dpde':
                var_out_ges[jv,j,:,:,:] = res.fl_gesges[:,:,:,-1]
                sname_ges.append('$f^2$')
            if vname == 'logL' or vname == 'dlogL' or vname == 'NLL':
                var_out_ges[jv,j,:,:,:] = res.logLG_gesges[:,:,:,-1]
                var_out_ges[jv,j,1,:,:] = res.logLebm_gesges[1,:,:,-1]
                var_out_ges[jv,j,3,:,:] = res.logLebm_gesges[3,:,:,-1]
                                
                if vname == 'dlogL':
                    for jm in range(Nmodel):
                        var_out_ges[jv,j,jm,:,:] = var_out_ges[jv,j,jm,:,:] - var_out_ges[jv,j,3,:,:]
                    sname_ges.append('dlogL')
                else:
                    sname_ges.append('logL')
                    
                if vname == 'NLL':
                    var_out_ges[jv, j, :,:,:] = -var_out_ges[jv, j, :,:,:]
                    sname_ges.pop()
                    sname_ges.append('NLL')
                
            if vname == 'dr' or vname == 'ddr':
                drbuf = np.zeros([Npar, Nrun])
                for jm in res.jmodel_vec:
                    for jN in range(Nrun):
                        for k in range(2):
                            rbuf = res.residuals_train_ges[jN][jm] if k == 0 else res.residuals_test_ges[jN][jm]
                            #drbuf[k, jN] = rbuf.max() - rbuf.min()
                            drbuf[k, jN] = rbuf.quantile(0.95) - rbuf.quantile(0.05)
                    var_out_ges[jv,j,jm,:,:] = drbuf
                
                if vname == 'ddr':
                    var_out_ges[jv,j,:,0,:] = var_out_ges[jv,j,:,1,:] - var_out_ges[jv,j,:,0,:]
                    sname_ges.append('$\Delta \Delta r$')
                else:
                    sname_ges.append('$\Delta r$')
                    
    for jv, k in enumerate(tt_list):
        var_out[jv,:,:,:] = var_out_ges[jv,:,:,k,:]
        if k == 1:
            if var_list[jv] != 'dpar_abs':
                sname_ges[jv] += ' validation'
            
    return var_out, sname_ges, res
        
sname_vec = ['logL validation','RMSE validation','$f^2$', '$|\Delta \lambda|$']      
#%%%%
#get names of models
def get_lvec():
    return ['PINN', 'PINN-EBM-off', 'PINN-off', 'PINN-EBM', 'NN', 'true']

#get name of jm-th model
def get_label(jm):
    lvec = get_lvec()
    return lvec[jm]

#get color for jm-th model
def get_color(jm):
    cvec = ['r','c','g','b','y','k']  
    return cvec[jm]

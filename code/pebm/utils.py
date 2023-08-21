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
        
        
#extract metrics of interest from stored run data
def gather_run_data(input_path0, fname0, Nrun, par_vec, par_label, var_list=['dpar_abs', 'rmse', 'dpde', 'logL']):
    Nmodel=5
    Npar=2
    rmse_ges, dpde_ges, dpar_ges, logL_ges = np.zeros([4, len(par_vec), Nmodel, Npar, Nrun])
    
    var_out_ges = np.zeros([len(var_list), len(par_vec), Nmodel, Npar, Nrun])
    
    fname = fname0 + str(Nrun)
    for j, par in enumerate(par_vec):
        input_path = input_path0 + par_label + str(par) + '/'
        filename = input_path + fname + '.dat'
        with open(filename,'rb') as f:
            res = pickle.load(f)
        
        for jv, vname in enumerate(var_list):
            if vname == 'dpar_abs':
                var_out_ges[jv,j,:,:,:] = res.dpargesges[:,:,:,-1]
                Ndpar = res.dpargesges.shape[1]
                for k in range(Ndpar):
                    var_out_ges[jv,j,:,k,:] = np.abs(var_out_ges[jv,j,:,k,:] - res.dpar[k])
            if vname == 'rmse':
                var_out_ges[jv,j,:,:,:] = res.rmse_gesges[:,:,:,-1]
            if vname == 'dpde':
                var_out_ges[jv,j,:,:,:] = res.fl_gesges[:,:,:,-1]
            if vname == 'logL' or vname == 'dlogL':
                var_out_ges[jv,j,:,:,:] = res.logLG_gesges[:,:,:,-1]
                var_out_ges[jv,j,1,:,:] = res.logLebm_gesges[1,:,:,-1]
                var_out_ges[jv,j,3,:,:] = res.logLebm_gesges[3,:,:,-1]
                                
                if vname == 'dlogL':
                    for jm in range(Nmodel):
                        var_out_ges[jv,j,jm,:,:] = var_out_ges[jv,j,jm,:,:] - var_out_ges[jv,j,3,:,:]
            
    return var_out_ges, res
        
        
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

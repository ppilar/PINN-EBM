# -*- coding: utf-8 -*-

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from pebm.plots import *
from pebm.results import *
from pebm.utils_train import calculate_stats, print_table, save_stats_to_txt


# input_path  = '../results/x1_main_results/x1_n3G/'
# fname = 'x1n3Gr10'
# step = 100


# input_path = '..//results/x3_main_results/x3_n3G/'
# fname = 'x3n3Gr10'
# step = 100

# input_path = '../results/lf2_x101_u/x101_nu_lf250/'
# fname = 'x101nur5'
# step = 500

input_path = '../results/x101_main_results/x101_n3G/'
fname = 'x101n3Gr5'
step = 500

plt.rcParams.update({'font.size': 14})

filename = input_path + fname + '.dat'
with open(filename,'rb') as f:
    res = pickle.load(f)           

if not hasattr(res, 'pars'):
    init_compatible_pars(res)
    res.pars['jmodel_vec'] = [0,3,2]                

#plot_dpar_statistics(res.dpargesges, res.dpar, res.Jgesges, res.lossdgesges, res.lossfgesges, res.Npar, input_path)
#plot_error_statistics(res.logLG_gesges, res.logLebm_gesges, res.rmse_gesges, input_path, res.itest)
if res.x_opt != 101:
    plot_teval_statistics(res.jmodel_vec, res.rmse_eval_gesges, res.fleval_gesges, res.teval, input_path)

save_stats_to_txt(res, fname, input_path)


#%%

model_vec = [3,2,0]
plot_learning_curves(res, input_path, fname, step=step, model_vec=model_vec)


#%%
# if res.x_opt != 101:
#     fig, ax = plt.subplots(1,2,figsize=(11,4))
#     axplot_statistics(ax[0], res.rmse_eval_gesges[:,0,:,:], xvec = res.teval.squeeze(), title='RMSE', xlabel='t', model_vec = model_vec)
#     axplot_statistics(ax[1], res.fleval_gesges[:,0,:,:], xvec = res.teval.squeeze(), title='PDE', xlabel='t', model_vec = model_vec)


# #%%   
if res.x_opt == 101:      
    plot_dpar_broken(res.dpargesges[:,:,:,::step], res.dpar, x_opt = res.x_opt, title = 'parameters $\lambda$', step=step, model_vec = model_vec)
    plt.savefig(input_path + 'pmet_lambdas_' + fname + '.pdf')
    
    
# #%%
# for jN in range(res.Nrun):
#     plot_residuals(res, jN=jN)
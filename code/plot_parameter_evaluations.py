# -*- coding: utf-8 -*-

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from pebm.plots import *
from pebm.results import *
from pebm.utils_train import calculate_stats, print_table


       

plt.rcParams.update({'font.size': 14})


#%%
#evaluate impact of weighting of pde loss
# input_path0 = '../results/lf2_x1/x1_n3G_'
# fname0 = 'x1n3Gr'
# lf_fac2_vec = [0.5,1,2,5,10,20,50]
# res = plot_par_eval(input_path0, fname0, 5, lf_fac2_vec, 'lf2', jmodel_vec = [3,2,0])
# plt.savefig('../results/plots/x1_omega.pdf')


#evaluate impact of number of training points
# N_train_vec = [20, 50, 100, 200]
# res = plot_par_eval('../results/x1_Ntr/x1_n3G_', 'x1n3Gr', 5, N_train_vec, 'Ntr', jmodel_vec = [3,2,0])
# plt.savefig('../results/plots/x1_Ntr.pdf')

#evaluate impact of noise strength
# fnoise_vec = [0.1, 0.2, 0.5, 0.75, 1, 2]
# res = plot_par_eval('../results/fn_x1_3G/x1_n3G_', 'x1n3Gr', 5, fnoise_vec, 'fn', jmodel_vec = [3,2,0])
# plt.savefig('../results/plots/x1_fn.pdf')


#%%
#evaluate impact of weighting of pde loss
# input_path0 = '../results/lf2_x101_3G/x101_n3G_'
# fname0 = 'x101n3Gr'
# lf_fac2_vec = [1,10,50,100]
# res = plot_par_eval(input_path0, fname0, 5, lf_fac2_vec, 'lf2', jmodel_vec = [3,2,0])
# plt.savefig('../results/plots/x101_omega.pdf')



#%% plot omega comparison for the different experiments
input_path0_vec = ['../results/x1_main_results/lf2_x1/x1_n3G_','../results/x3_main_results/lf2_x3/x3_n3G_', '../results/x101_main_results/lf2_x101/x101_n3G_']
fname0_vec = ['x1n3Gr','x3n3Gr','x101n3Gr']


mname_vec = ['exponential', 'Bessel', 'Navier Stokes']
par_label = 'lf2'
par_label2 = '$\omega$'


fig, axs = plt.subplots(4,3,figsize=(14,17))
fig.tight_layout(h_pad=3)
for j, input_path0 in enumerate(input_path0_vec):
    if j != 5:
        fname0 = fname0_vec[j]
        if j == 0:
            par_vec = [0.5,1,2,5,10,20,50]
        if j == 1:
            par_vec = [0.5,1,2,5,10,20,50]
        if j == 2:
            par_vec = [1, 10, 50, 100]
            
        stat_vec, sname_vec, res = gather_run_data(input_path0, fname0, 5, par_vec, par_label, var_list=['NLL', 'rmse', 'dpde', 'dpar_abs'], tt_list = [1,1,0,0])
        
        
        sname_vec[0] = mname_vec[j] + '\n\n' + sname_vec[0]
        for k in range(4):            
            jmodel_vec = [3,2,0]
            axplot_stat(axs[k,j], stat_vec[k], par_vec, sname_vec[k], par_label2, jmodel_vec = jmodel_vec)
   
plt.savefig('../results/plots/omega_plot.pdf')







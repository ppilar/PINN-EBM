# -*- coding: utf-8 -*-
#use this file to loop over various settings

import os
import sys
import random
import torch
import numpy as np
from pebm.utils import *

def stop_execution():
    print('no parameter values specified!')
    sys.exit()

rand_init, s = init_random_seeds(s=0)
path0 = '../results/' #folder where results will be saved

pars = dict()
set_par(pars, 'jmodel_vec', [0, 3, 2]) #which models to include in runs
set_par(pars, 'x_opt', 1) #pick differential equation
par_label = 'fn'  #pick parameter to loop over (n, lf2 (=omega), Ntr, fn)


if pars['x_opt'] == 1: #exponential function
    if par_label == 'n':
        pars['Nrun'] = 10
        pars['Npinn'] = 40000        
        pars['lf_fac2'] = 1
        par_vec = ['G','u', '3G', '3G0']
    elif par_label == 'lf2':
        pars['Nrun'] = 5
        pars['Npinn'] = 50000        
        pars['n_opt'] = '3G'
        par_vec = [1,2,5,10,20,50]
    elif par_label == 'Ntr':
        pars['Nrun'] = 5
        pars['Npinn'] = 50000 
        pars['n_opt'] = '3G'
        pars['lf_fac2'] = 1
        par_vec = [20, 50, 100, 200]
    elif par_label == 'fn':
        pars['Nrun'] = 5
        pars['Npinn'] = 50000      
        pars['n_opt'] = '3G'
        pars['lf_fac2'] = 1
        #par_vec = [0.1, 0.25, 0.5]
        par_vec = [0.2, 0.5, 0.75, 1, 2]
    else:
        stop_execution()
elif pars['x_opt'] == 3: #Bessel function
    if par_label == 'n':
        pars['lf_fac2'] = 1
        pars['Nrun'] = 10
        pars['Npinn'] = 100000
        par_vec = ['3G']
    elif par_label == 'lf2':
        pars['Nrun'] = 5
        pars['Npinn'] = 50000
        pars['n_opt'] = '3G'
        par_vec = [1,2,5,10,20,50]
    elif par_label == 'Ntr':
        pars['Nrun'] = 5
        pars['Npinn'] = 50000 
        pars['n_opt'] = '3G'
        pars['lf_fac2'] = 1
        par_vec = [20, 50, 100, 200]
    elif par_label == 'fn':
        pars['Nrun'] = 5
        pars['Npinn'] = 50000      
        pars['n_opt'] = '3G'
        pars['lf_fac2'] = 1
        #par_vec = [0.1, 0.25, 0.5]
        par_vec = [0.1, 0.2, 0.5, 0.75, 1, 2]
    else:
        stop_execution()
elif pars['x_opt'] == 101: #Navier Stokes
    pars['Nrun'] = 5
    if par_label == 'n':
        pars['lf_fac2'] = 50 # omega = 50 better for PINN-EBM
        pars['lf_fac2_alt'] = 1 # PINN and PINN-off better with omega=1
        pars['Npinn'] = 100000
        par_vec = ['3G']
    elif par_label == 'lf2':
        pars['n_opt'] = 'u'  
        pars['Npinn'] = 50000
        par_vec = [1,10,50,100]
    else:
        stop_execution()
    
    
    
#run
for j, par in enumerate(par_vec):
    if par_label == 'lf2':
        pars['lf_fac2'] = par
    elif par_label == 'Ntr':
        pars['N_train'] = par
    elif par_label == 'n':
        pars['n_opt'] = par   
    elif par_label == 'fn':
        pars['fnoise'] = par
        
    folder0 = 'x' + str(pars['x_opt']) + '_n' + pars['n_opt']
    if par_label != 'n':
        folder0 += '_' + par_label + str(par)
    input_path = set_input_path(path0, folder0)
     
    exec(open('pebm.py').read())


    

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


jmodel_vec = [0,3,2] #which models to include in runs
x_opt = 101 #pick differential equation
par_label = 'n'  #pick parameter to loop over (n, lf2 (=omega), Ntr)


if x_opt == 1: #exponential function
    if par_label == 'n':
        Nrun = 10
        Npinn = 40000        
        lf_fac2 = 1
        par_vec = ['G','u', '3G', '3G0']
    elif par_label == 'lf2':
        Nrun = 5
        Npinn = 50000        
        n_opt = '3G'
        par_vec = [1,2,5,10,20,50]
    elif par_label == 'Ntr':
        Nrun = 5
        Npinn = 50000 
        n_opt = '3G'
        lf_fac2 = 1
        par_vec = [20, 50, 100, 200]
    else:
        stop_execution()
elif x_opt == 3: #Bessel function
    if par_label == 'n':
        lf_fac2 = 1
        Nrun = 10
        Npinn = 100000
        par_vec = ['3G']
    elif par_label == 'lf2':
        Nrun = 5
        Npinn = 50000
        n_opt = '3G'
        par_vec = [1,2,5,10,20,50]
    else:
        stop_execution()
elif x_opt == 101: #Navier Stokes
    Nrun = 5
    if par_label == 'n':
        lf_fac2 = 50 # omega = 50 better for PINN-EBM
        lf_fac2_alt = 1 # PINN and PINN-off better with omega=1
        Npinn = 100000
        par_vec = ['3G']
    elif par_label == 'lf2':
        n_opt = '3G'  
        Npinn = 50000
        par_vec = [1,10,50,100]
    else:
        stop_execution()
    
    
    
#run
for j, par in enumerate(par_vec):
    if par_label == 'lf2':
        lf_fac2 = par
    elif par_label == 'Ntr':
        N_train = par
    elif par_label == 'n':
        n_opt = par   
        
    folder0 = 'x' + str(x_opt) + '_n' + n_opt
    if par_label != 'n':
        folder0 += '_' + par_label + str(par)
    input_path = set_input_path(path0, folder0)
     
    exec(open('pebm.py').read())


    

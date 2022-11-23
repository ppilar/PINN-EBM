# -*- coding: utf-8 -*-
#use this file to loop over various settings

import os
from pebm.utils_train import check_dirs


path0 = '../results/'
x_opt = 1
n_opt_vec = ['G','u', '3G', '3G0']

for j, n_opt in enumerate(n_opt_vec):    
    folder0 = 'x' + str(x_opt) + '_n' + n_opt    
    input_path = path0 + folder0 + '/'
    check_dirs(path0, input_path)
    
    print(input_path)
    exec(open('pebm.py').read())

if not 'pars' in locals(): pars = dict()

set_par(pars, 'x_opt', 1) #  choose pde: 1 ... exp_1d, 2 ... sin_1d, 3 ... bessel_1d, 4 ... sin_2d, 5 ... sin_exp_2d, 6 ... exp_2d, 101 ... navier_stokes
set_par(pars, 'n_opt', '3G')  #choose noise type; 'G', 'sG', 'u', '3G', '3G0', 'Gmix', 'rmix', 't'

set_par(pars, 'Npinn', 50000) #how many iterations to train for
set_par(pars, 'Nrun', 1) #number of runs to average over
set_par(pars, 'jmodel_vec', [0,3,2])  # 0 ... pinn0, 1 ... pinn-ebm, 2 ... pinn-offset, 3 ... pinn0-ebm, 4 .. G_mixture


buf = 4000 if pars['x_opt'] != 101 else 10000 #at what iteration to start using ebm
set_par(pars, 'i_init_ebm', buf)
buf = int(1*pars['Npinn']) if pars['x_opt'] != 101 else int(0.8*pars['Npinn'])  #after how many iterations to take scheduler step
set_par(pars, 'i_sched', buf)
buf = 100 if pars['x_opt'] != 101 else 500 #after how many iterations to (repeatedly) calculate statistics on test data
set_par(pars, 'itest', buf)
set_par(pars, 'iplot', 25000) #after how many iterations to make plot of current predictions

## init values
#set_par(pars, 'dpar','')
set_par(pars, 'fnoise', 1)
#set_par(pars, 'N_train', 10)
ds = get_ds(pars)
set_par(pars, 'lf_fac2', 1)#1)
set_par(pars, 'lf_fac2_alt', -1)
set_par(pars, 'ebm_ubound', 1)


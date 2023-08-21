# define setting
if not 'x_opt' in locals(): x_opt = 1 #  choose pde: 1 ... exp_1d, 2 ... sin_1d, 3 ... bessel_1d, 4 ... sin_2d, 5 ... sin_exp_2d, 6 ... exp_2d, 101 ... navier_stokes
if not 'n_opt' in locals(): n_opt = '3G'  #choose noise type; 'G', 'sG', 'u', '3G', '3G0', 'Gmix', 'rmix', 't'

if not 'Npinn' in locals(): Npinn = 50000 #how many iterations to train for 
if not 'Nrun' in locals(): Nrun = 5 #number of runs to average over
if not 'jmodel_vec' in locals(): jmodel_vec = [0,3,2,1]  # 0 ... pinn0, 1 ... pinn-ebm, 2 ... pinn-offset, 3 ... pinn0-ebm, 4 .. G_mixture


i_init_ebm = 4000 if x_opt != 101 else 10000 #at what iteration to start using ebm
i_sched = int(1*Npinn) if x_opt != 101 else int(0.8*Npinn)  #after how many iterations to take scheduler step
#i_lf_fac = i_init_ebm #after how many iterations to use higher weighting factor for pde loss
itest = 100 if x_opt != 101 else 500 #after how many iterations to (repeatedly) calculate statistics on test data
iplot = 25000 #after how many iterations to make plot of current predictions

## init values
if not 'dpar' in locals(): dpar = ''
ds = get_ds(x_opt, dpar)
batch_size_train, batch_size_coll, lr_pinn, lr_ebm, ld_fac, lf_fac, Nebm, i_init_ebm, Npinn = ds.init_train_pars(Npinn = Npinn, i_init_ebm = i_init_ebm)
tmin, tmax, tmin_coll, tmax_coll, n_fac, N_train0, N_coll = ds.init_data_ranges()
Uvec_pinn, Uvec_ebm, fdrop_pinn, fdrop_ebm = ds.init_network_pars()
if not 'N_train' in locals(): N_train = N_train0
if not 'lf_fac2' in locals(): lf_fac2 = 1
if not 'ebm_ubound' in locals(): ebm_ubound = 1

#Nebm = 5000
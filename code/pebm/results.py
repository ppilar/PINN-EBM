# -*- coding: utf-8 -*-

import time
import torch
import numpy as np

#class to store results
class Results():
    def __init__(self, x_opt, n_opt, Npinn, Nrun, jmodel_vec, dpar, itest, N_train, lf_fac2, ebm_ubound, Nmodel = 5):
        self.x_opt = x_opt
        self.n_opt = n_opt
        self.Nmodel = Nmodel
        self.Npinn = Npinn
        self.Nrun = Nrun
        self.jmodel_vec = jmodel_vec
        self.Npar = len(dpar)
        self.dpar = dpar
        self.itest = itest        
        
        #add other training parameters
        self.dpargesges = np.zeros([Nmodel, self.Npar, Nrun, Npinn])
        self.Jgesges, self.lossdgesges, self.lossfgesges = np.zeros([3, Nmodel, Nrun, Npinn])
        self.logLG_gesges, self.logLebm_gesges, self.rmse_gesges, self.fl_gesges = np.zeros([4, Nmodel, 2, Nrun, Npinn//itest])
        self.tm_gesges, self.tebm_gesges = np.zeros([2, Nmodel, Nrun])
        self.ebm_curve_ges = []
       
        self.Nteval = 501
        self.rmse_eval_gesges = np.zeros((Nmodel,1,Nrun,self.Nteval))
        self.fleval_gesges = np.zeros((Nmodel,1,Nrun,self.Nteval))
        
    #initialize empty lists and arrays for sotring results of current run
    def init_run_results(self):
        self.tebm_avg, self.tzero_avg = [0, 0]
        self.tpinn_avg = np.zeros((self.Nmodel, self.Nrun))
        self.Jebm = []
        self.tm_ges, self.tebm_ges = np.zeros([2, self.Nmodel])
        self.lossdges, self.lossfges, self.Jges = [([],[],[],[],[],[]) for j in range(3)]
        self.logLG_ges, self.logLebm_ges, self.rmse_ges, self.fl_ges = [(([],[]),([],[]),([],[]),([],[]),([],[])) for j in range(4)]  
        self.dparges = tuple([[[] for j in range(self.Npar)] for i in range(self.Nmodel)])

    #store results of current run
    def store_run_results(self, jm, jN):
        for jp in range(self.Npar):     
            self.dpargesges[jm,jp,jN,:] = np.array(self.dparges[jm][jp][:])                
        self.Jgesges[jm,jN,:] = np.array(self.Jges[jm])
        self.lossdgesges[jm,jN,:] = np.array(self.lossdges[jm])
        self.lossfgesges[jm,jN,:] = np.array(self.lossfges[jm])
        self.logLG_gesges[jm,0,jN,:] = np.array(self.logLG_ges[jm][0]) #train
        self.logLG_gesges[jm,1,jN,:] = np.array(self.logLG_ges[jm][1]) #validation
        self.logLebm_gesges[jm,0,jN,:] = np.array(self.logLebm_ges[jm][0]) #train
        self.logLebm_gesges[jm,1,jN,:] = np.array(self.logLebm_ges[jm][1]) #validation
        self.rmse_gesges[jm,0,jN,:] = np.array(self.rmse_ges[jm][0]) #train
        self.rmse_gesges[jm,1,jN,:] = np.array(self.rmse_ges[jm][1]) #validation
        self.fl_gesges[jm,0,jN,:] = np.array(self.fl_ges[jm][0]) #train
        self.fl_gesges[jm,1,jN,:] = np.array(self.fl_ges[jm][1]) #validation
        
        if self.x_opt != 101:
            self.fleval_gesges[jm,0,jN,:] = self.fleval
            self.rmse_eval_gesges[jm,0,jN,:] = self.rmse_eval
            
        self.tm_gesges[jm, jN] = self.tm_ges[jm]
        self.tebm_gesges[jm, jN] = self.tebm_ges[jm]
        

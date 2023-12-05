# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
from .utils_train import Net, get_residuals

import torch.nn as nn
import torch.optim as optim

class EBM():
    #initialize cEBM
    def __init__(self, pars, ymax, device):
        self.ebm_ubound = pars['ebm_ubound']
        self.Uvec_ebm = pars['Uvec_ebm']
        self.fdrop_ebm = pars['fdrop_ebm']
        self.lr_ebm = pars['lr_ebm']
        self.bs_train = pars['bs_train']
        self.Nebm = pars['Nebm']
        
        self.device = device
        self.init = 0
        self.indicator = 1000
        
        #initialize dummies
        self.ebm_bounds = (-ymax.item(), ymax.item(), 0, 10)
        self.net_ebm = Net(self.Uvec_ebm, fdrop=self.fdrop_ebm, bounds = self.ebm_bounds).to(device)
        self.optimizer_ebm = optim.Adam(self.net_ebm.parameters(), lr=self.lr_ebm)
        
        
    #forward through net, if called
    def __call__(self, x):
        return self.net_ebm(x)
        
    #initialize network parameters and miscallaneous values
    def initialize(self, residuals, Jebm, tebm_avg):
        
        res_max = torch.max(torch.abs(residuals.min()), torch.abs(residuals.max()))
        self.ebm_bounds = (-res_max.item(), res_max.item(), 0, self.ebm_ubound)
        self.net_ebm = Net(self.Uvec_ebm, fdrop=self.fdrop_ebm, bounds = self.ebm_bounds).to(self.device)
        self.optimizer_ebm = optim.Adam(self.net_ebm.parameters(), lr=self.lr_ebm)
        self.scheduler_ebm = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_ebm, gamma=0.3)
        self.init = 1
        tebm_avg, Jebm = self.train_ebm(residuals, Jebm, tebm_avg)
        return tebm_avg, Jebm
        
    #initialize the EBM
    def train_ebm(self, residuals, Jebm, tebm_avg):
        print('init_ebm!')
        self.net_ebm.train()
        
        ebm_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(residuals), batch_size=self.bs_train, num_workers=0,  shuffle=True) 
        
        tebm = time.time()
        it_ebm = 0
        j = 0
        while(True):
            for i, batch in enumerate(ebm_loader):
                res_batch = batch[0]
                
                J = self.get_mean_NLL(res_batch)
                
                self.optimizer_ebm.zero_grad()
                J.backward()
                self.optimizer_ebm.step()
                Jebm.append(J.item())
                it_ebm += 1
                
                if it_ebm == self.Nebm:
                    break
                
            if j > 0: tebm_avg = (tebm_avg*(j) + time.time() - tebm)/(j+1)        
            print("\r", 'j=' + str(j) + ', it=' + str(it_ebm), end="")
            j += 1
            if it_ebm == self.Nebm:
                break
        
        self.net_ebm.eval()
        self.set_pdf_ges(residuals)
        self.set_indicator()
            
        self.net_ebm.train()
        return tebm_avg, Jebm
    
    #get linspace over entire range of residual values, over which PDF will be normalized
    def get_rvec(self, res):
        res = res.flatten().unsqueeze(1)    
        Nres = res.shape[0]
        lb = (res.min() - 5*res.std()).detach()
        ub = (res.max() + 5*res.std()).detach()    
        rvec = torch.linspace(lb, ub, 1001, device=self.device) 
        return rvec
        
    #calculate PDF and Z
    def set_pdf_ges(self, residuals_ges):
        self.rvec_ges = self.get_rvec(residuals_ges)
        self.set_Z()
        self.pdf_ges = self.get_pdf(self.rvec_ges).detach().cpu()
    
    #calculate indicator to determine if EBM has been successfully trained (i.e. PDF=0 away from data)
    def set_indicator(self):
        pdf_max = self.pdf_ges.max()
        ind1 = self.pdf_ges[0]
        ind2 = self.pdf_ges[-1]
        self.indicator = torch.maximum(ind1,ind2)
        self.thr = pdf_max/300
    
    #calculate mean negative log likelihood (NLL)
    def get_mean_NLL(self, res):
        Nres = res.shape[0]
        rvec = self.get_rvec(res)
        
        buf = self.net_ebm(rvec.unsqueeze(1))
        mbuf = buf.max()
        buf_Z = torch.exp(buf-mbuf).squeeze()
        buf_res = self.net_ebm(res.flatten().unsqueeze(1))
        
        
        Z = torch.trapezoid(buf_Z, rvec)
        J = -torch.sum(buf_res) + Nres*torch.log(Z) + Nres*mbuf.squeeze()
        J2 = J/Nres
        
        #print('lb:',str(lb.item()),'ub:',str(ub.item()),'buf_Z:',str(buf_Z.max().item()),'Z:',str(Z.item()))
        return J2
    
    #update partition function Z
    def set_Z(self):
        buf = torch.exp(self.net_ebm(self.rvec_ges.flatten().unsqueeze(1))).squeeze()
        self.Z = torch.trapezoid(buf, self.rvec_ges)
    
    #get PDF on rvec
    def get_pdf(self, rvec):
        self.set_Z()
        buf = torch.exp(self.net_ebm(rvec.flatten().unsqueeze(1))).squeeze()
        return (buf/self.Z).detach().cpu()
    
    
    
    
    
    
    
    
    
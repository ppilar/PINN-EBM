# -*- coding: utf-8 -*-
import numpy as np
import torch
import random

from scipy.stats import skewnorm
from scipy.special import erf

def get_noise(n_opt, f, pars=0):
    #load desired noise function
    
    if n_opt == 'G': #Gaussian
        return n_G(n_opt, f, pars)
    if n_opt == 'Goff': #Gaussian with non-zero mean
        return n_G(n_opt, f, [5,2.5])
    if n_opt == 'sG': #skewed Gaussian
        return n_sG(n_opt, f, pars)
    if n_opt == 'u': #uniform
        return n_uniform(n_opt, f, pars)
    if n_opt == '3G': #mixture of Gaussians
        par_list = [[0.0, 2],[4.0,4],[8.0,0.5]]
        n_list = ['G']*3
        pi_list = [1/3]*3
        return n_mixture(n_opt, f, par_list, n_list, pi_list)
    if n_opt == '3G0': #mixture of Gaussians with zero mean
        par_list = [[-4.0, 2],[0.,4],[4.0,0.5]]
        n_list = ['G']*3
        pi_list = [1/3]*3
        return n_mixture(n_opt, f, par_list, n_list, pi_list)    
    if n_opt == 'mix0': #mixture of Gaussian and uniforms
        n_list = ['G', 'u', 'u']
        par_list = [[0.0, 2], [1,7], [0, 12]]
        pi_list = [1/3]*3
        return n_mixture(n_opt, f, par_list, n_list, pi_list)
    if n_opt == 'Gmix': #random Gaussian mixture
        n_list = ['G']*3
        pi_list = [1/3]*3
        return n_mixture(n_opt, f, -1, n_list, pi_list)
    if n_opt == 'rmix': #random mixture
        return n_mixture(n_opt, f, -1)
    if n_opt == 'sGtest': #specific skewed Gaussian
        return n_sG(n_opt, 1, pars=[18.8, -2.3, 4.6])
        #return n_sG(n_opt, 1, pars=[1, -10.3, 1])
    print('error! noise not defined!')
    
    
class Noise():
    def __init__(self, n_opt, f, pars=[]):
        self.n_opt = n_opt
        self.f = f
        self.pars = pars
        
    def sample(self, Ns):
        #allows to sample from noise distribution
        print('not implemented!')
        
    def pdf(self, x):
        #returns the noise pdf(x)
        print('not implemented!')
        
    def init_pars(self):
        #initializes the noise distribution either with the argument pars, if given, or with standard values
        print('not implemented!')

class n_G(Noise):
    #Gaussian noise
    
    def __init__(self, n_opt, f, pars):
        if type(pars) == int: pars = self.init_pars(pars)
        Noise.__init__(self, n_opt, f, pars)
        self.mu, self.sig = pars
        
    def sample(self, Ns):
        return torch.tensor(np.random.normal(self.mu*self.f, self.sig*self.f, Ns))   #4 for exp
    
    def pdf(self, x):
        q = torch.distributions.normal.Normal(self.mu*self.f,self.sig*self.f)
        pdf = torch.exp(q.log_prob(x))
        return pdf
    
    def init_pars(self, pars):
        if pars == 0:
            pars = [0,2.5]      
        else:
            pars = []
            pars.append(5*(torch.rand(1)-1).item()) #mu
            pars.append(4*torch.abs(torch.rand(1)).item()) #sig
        return pars
    
class n_sG(Noise):
    #skewed Gaussian noise
    
    def __init__(self, n_opt, f, pars):
        if type(pars) == int: pars = self.init_pars(pars)
        Noise.__init__(self, n_opt, f, pars)
        self.a, self.xi, self.w = pars
        self.m = get_sG_m(torch.tensor(self.a), torch.tensor(self.xi), torch.tensor(self.w)).item()
        #self.m = self.m/self.xi
        
    def sample(self, Ns):
        #return (torch.tensor(skewnorm.rvs(self.a,scale=self.w,size=Ns) + self.xi - self.m))*self.f
        return (torch.tensor(skewnorm.rvs(self.a,scale=self.w,size=Ns) + self.xi))*self.f
    
    def pdf(self, x):
        pdf = get_sG_pdf(x/self.f, self.a, self.xi, self.w)
        return pdf
    
    def init_pars(self, pars):
        if pars == 0:
            pars =  [10, -1, 4]       
        else:
            pars = []
            pars.append(torch.abs(10 + 10*torch.rand(1)).item()) #a
            pars.append(0 + 5*(torch.rand(1)-1).item()) #xi
            pars.append(torch.abs(4 + 4*torch.rand(1)).item()) #w
        return pars
    
class n_uniform(Noise):
    #uniform noise
    
    def __init__(self, n_opt, f, pars):
        if type(pars) == int:  pars = self.init_pars(pars)
        Noise.__init__(self, n_opt, f, pars)
        self.a, self.b = pars
        
    def sample(self, Ns):
        return (self.b - self.a)*self.f*torch.rand(Ns) + self.a*self.f
    
    def pdf(self, x):
        pdf = 1/(self.f*(self.b - self.a))*torch.ones(x.shape)  
        iz = torch.cat((torch.where(x < self.a*self.f)[0], torch.where(x > self.b*self.f)[0]))
        pdf[iz] = 0
        return pdf
    
    def init_pars(self, pars):
        if pars == 0:
            pars =  [0, 10]      
        else:
            pars = []
            pars.append((0 + 5*(torch.rand(1))).item()) #a
            pars.append((7.5 + 5*(torch.rand(1))).item()) #b
        return pars
    
class n_mixture(Noise):
    #mixture of other noise distributions
    
    def __init__(self, n_opt, f_list, par_list, n_list = [], pi_list = [], Nc=3):        
        if type(par_list) == int:
            par_list, n_list, pi_list = self.init_pars(par_list, n_list, pi_list, Nc)
        
        Noise.__init__(self, n_opt, f_list, par_list)
        
        
        self.Nc = len(n_list)   
        self.f_list = [f_list]*self.Nc if (type(f_list) in [float, int]) else f_list
        self.n_list = n_list
        self.par_list = par_list
        self.pis = torch.tensor(pi_list)
        
        self.dists = []
        for j in range(self.Nc):
            self.dists.append(get_noise(self.n_list[j], self.f_list[j], self.par_list[j]))
        
    def sample(self, Ns):
        inds = torch.multinomial(self.pis, Ns, replacement=True)
        samples = torch.zeros(Ns)
        Nj0 = 0
        for j in range(self.Nc):
            Nj = (inds==j).sum().item()
            samples[Nj0:Nj0+Nj] = self.dists[j].sample([Nj])
            Nj0 += Nj
        
        ishuffle = torch.randperm(Ns)
        return samples[ishuffle]
    
    def pdf(self, x):
        pdf = torch.zeros(x.shape[0])
        for j in range(self.Nc):
            pdf += self.pis[j]*self.dists[j].pdf(x)
        return pdf
    
    def init_pars(self, par_list, n_list, pi_list, Nc):
        if par_list == 0:
            par_list = [[0.0, 2],[4.0,4],[8.0,0.5]]
            n_list = ['G']*3
            pi_list = [1/3]*3            
        else:        
            if len(n_list) == 0:
                n_opts = ['G', 'u', 'sG']
                n_list = random.choices(n_opts,k=Nc)
            if len(pi_list) == 0:
                pibuf = torch.abs(torch.randn(Nc))
                pi_list = pibuf/torch.sum(pibuf)
            
            par_list = []
            dists = []
            for j in range(Nc):
                dists.append(get_noise(n_list[j], 1, -1))
                par_list.append(dists[j].pars)                          
        
        return par_list, n_list, pi_list
        
    
    
    
###################################
###################################
###################################
###################################

        


def sG_pdf(x, a=10, xi=0,w=4): 
    #skewed Gaussian pdf
    
    #pars
    m = get_sG_m(torch.tensor(a), torch.tensor(xi), torch.tensor(w)).numpy()
    
    #calculation
    x = x
    pdf = 2/(w*np.sqrt(2*np.pi))*np.exp(-(x-xi)**2/(2*w**2))
    f =  1/np.sqrt(2*np.pi)*np.sqrt(2)*np.sqrt(np.pi)/2
    ub = a*(x-xi)/w*1/np.sqrt(2)
    if x >= 0:
        buf = (1 + erf(ub))
    else:        
        buf = (1 - erf(-ub))
    return pdf*f*buf

def get_sG_pdf(xvec, a=10, xi=0, w=4):
    #return function of skewed Gaussian pdf
    
    N = xvec.shape[0]
    pdf2 = np.zeros(N)
    for j in range(N):
        pdf2[j] = sG_pdf(xvec[j], a, xi, w)
    return pdf2
    
def get_sG_m(a, xi, w):
    #returns (approximate) mean of skewed Gaussian
    
    v = torch.tensor(2.)
    delta = a/torch.sqrt(1+a**2)
    mu = xi + w*delta*torch.sqrt(v/torch.pi)
    g1 = (4-torch.pi)/2*(delta*torch.sqrt(v/torch.pi))**3/(1-2*delta**2/torch.pi)**(3/2)
    muz =  torch.sqrt(v/torch.pi)
    m0 = muz - g1*torch.sqrt(1-muz**2)/2 - torch.sign(a)/2*torch.exp(-2*torch.pi/torch.abs(a))
    m = xi + w*m0
    return m
            


        
        
        
        
        
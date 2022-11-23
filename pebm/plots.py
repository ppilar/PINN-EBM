# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:41:38 2022

@author: phipi206
"""

import matplotlib.pyplot as plt
import torch
import numpy as np



def plot_data(dim, t_train, y_train, t_test, y_test, f_train):
    #plot training and test data
    
    N = f_train.shape[0]
    sN = int(np.sqrt(N))
    if dim == 1:
        plt.scatter(t_train, y_train)
        plt.scatter(t_test, y_test, c = 'r')
    if dim == 2:
        fig = plt.figure(figsize = (12,4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        scatter_plot = ax.scatter3D(t_train[:,0], t_train[:,1], y_train, marker='o')
        ax.scatter3D(t_test[:,0], t_test[:,1], y_test, marker='o', c='r')
        
        
        mycmap = plt.get_cmap('gist_earth')
        buf1, buf2 = t_train[:,0].reshape((sN,sN)), t_train[:,1].reshape((sN,sN))
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')        
        surf = ax1.plot_surface(buf1.numpy(), buf2.numpy(), f_train.reshape((sN,sN)).numpy(), cmap=mycmap)
        
    plt.show()




def comparison_plot(model, t_train, y_train, t_plot, ages, path):
    #compare predictions of different pinns and true curve
    
    x_plot = model.x(t_plot)
    y_plot = model.y(x_plot)
    x_net_plot = model.net(t_plot.unsqueeze(1))
    x_net0_plot = model.net0(t_plot.unsqueeze(1))

    fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(12,4))    

    for j in range(2):
        if j == 0:
            ax = ax0
            jm = -1
        else:
            ax = ax1
            jm = 100
        ax.plot(t_plot[:jm], x_plot[:jm], label='x_true')
        ax.plot(t_plot[:jm], x_net0_plot.detach()[:jm],'--', label='x_pinn0')
        ax.plot(t_plot[:jm], x_net_plot.detach()[:jm],'--', label='x_pinn')
        ax.plot(t_plot[:jm], y_plot[:jm], label='y_true')
        ax.scatter(t_train.detach(), y_train, marker='x',c='r',s=10, label='y_meas')
        ax.set_xlabel('t')
        
        if j == 0:
            ax.axvline(x=15,linestyle=':',c='grey', label='coll')
            ax.set_ylim([0,150])
        else:
            ax.set_ylim([0,30])

        #add legend to plot
        handles, labels = ax.get_legend_handles_labels()
        order = [0,1,2,4,3]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    next(ax2._get_lines.prop_cycler)['color']
    ax2.plot(ages[0])
    ax2.plot(ages[1])
    ax2.legend(['a0->{:.{}f}'.format(ages[0][-100:].mean().item(),3),'a->{:.{}f}'.format(ages[1][-100:].mean().item(),3)])
    

    plt.savefig(path + '.pdf')


def plot_ebm(net_ebm, res, opar, noise, ymin, ymax, path, device, bins=21):
    #plot pdf learned by EBM, together with true one
    
    res = res.detach().to('cpu').numpy()
    opar = opar.detach().to('cpu').numpy()

    net_train = 0    
    if net_ebm.training == True:    
        net_ebm.eval()
        net_train = 1
        
    
    vvec = [net_ebm]
    switch_device(vvec)
    
    pmin = min(np.min(res), ymin)
    pmax = max(np.max(res), ymax)
    y_plot = torch.linspace(pmin - 0.2*np.abs(pmin) + opar.item(),pmax + 0.2*np.abs(pmax) + opar.item(),201)
    p_net = net_ebm((y_plot-opar).unsqueeze(1)).detach().numpy().squeeze()    
    y_net = np.exp(p_net)
    
    y_true = noise.pdf(y_plot)
    indn = np.argmax(y_net)
    indt = np.argmax(y_true)
    
    plt.figure()
    y, x, _ = plt.hist((res+opar).squeeze(), bins=bins, density=True, color='g')
    f = np.max(y)
    plt.plot(y_plot, y_net/y_net[indn]*f, color='b')
    plt.plot(y_plot, y_true/y_true[indt]*f, color='k')
    plt.xlabel('$\hat x + \theta_0 - y$')
    plt.legend(['ebm', 'true'])
    plt.savefig(path + 'ebm.pdf')
    
    switch_device(vvec, device)
    
    if net_train == 1:
        net_ebm.train()
        
    plt.show()
    

def plot_ebm_comp(y_plot, curves_ebm, curve_true, path):
    #plot ebm predictions of multiple runs together
    
    y_plot = y_plot.cpu().numpy()
    
    curves_ebm_norm = np.zeros([len(curves_ebm), y_plot.shape[0]])
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(y_plot, curve_true.cpu().numpy(), color = 'k', label = 'true')
    for j in range(len(curves_ebm)):
        buf = curves_ebm[j].numpy()
        norm = np.trapz(buf, y_plot)
        curves_ebm_norm[j] = buf/norm
        ax[0].plot(y_plot, curves_ebm_norm[j] , 'b--', label = 'ebm')
    
    ax[0].legend()
    
    ax[1].plot(y_plot, curve_true.cpu().numpy(), color = 'k', label = 'true')
    mu = np.mean(curves_ebm_norm,0)
    std = np.std(curves_ebm_norm,0)
    ax[1].plot(y_plot, mu, color='b', label='ebm')
    ax[1].fill_between(y_plot,mu + std, mu - std, color='b', alpha=0.5)
    ax[1].legend()
    plt.savefig(path + 'ebm_comp.pdf')
    plt.show()

    
def switch_device(vvec, device = 'cpu'):
    for v in vvec:
        v.to(device)


def plot_pinn(net_pinn, fx, t_train, y_train, t_test, y_test, tmin, tmax, tmin_coll, tmax_coll, path, device):
    #plot pinn prediction
    
    dim = t_train.shape[1]
    
    if dim == 1:
        plot_pinn_1d(net_pinn, fx, t_train, y_train, t_test, y_test, tmin_coll[0], tmax_coll[0], path, device)
    elif dim == 2:
        plot_pinn_2d(net_pinn, fx, t_train, y_train, t_test, y_test, tmin, tmax, tmin_coll, tmax_coll, path, device)

def plot_pinn_1d(net_pinn, fx, t_train, y_train, t_test, y_test, tmin, tmax, path, device):
    #plot pinn prediction with 1D input
    
    net_train = 0   
    np0 = 0
    
    N = len(net_pinn)        
    vvec = [t_train, y_train]
    switch_device(vvec)
            
    t_plot = torch.linspace(tmin,tmax,200)
    x_plot = fx(t_plot) 
    t_plot = t_plot.to(device)
    
    xnp_ges = []
    for netp in net_pinn:
        netp.eval().to(device)
        #no measurement equation, since we are interested in latent function
        xnp_ges.append(netp(t_plot.unsqueeze(1)).detach().cpu().numpy().squeeze())
        netp.to('cpu')
        
    t_plot = t_plot.to('cpu')
    if N > 1:
        cvec = ['r','b','g','c']
    else:
        cvec = ['b']
        
    plt.figure()
    for xnp, j in zip(xnp_ges, range(N)):
        plt.plot(t_plot, xnp, color=cvec[j])
        
    if N == 4:
        lvec = ['pinn0', 'pinn-ebm', 'pinn-off', 'pinn0-ebm', 'true']
    else:
        lvec = ['pinn-ebm', 'true']
        
        
    plt.plot(t_plot, x_plot,color='k')
    plt.scatter(t_train.detach().cpu().numpy(), y_train.cpu().numpy(),s=4)
    plt.scatter(t_test.detach().cpu().numpy(), y_test.cpu().numpy(), s=4, c='r')
    plt.legend(lvec)
    plt.savefig(path + 'pinn.pdf')
    
    
    
    switch_device(vvec, device)
    if N == 1:
        net_pinn[0].to(device).train()
        
    plt.show()

def plot_pinn_2d(net_pinn, fx, t_train, y_train, t_test, y_test, tmin, tmax, tmin_coll, tmax_coll, path, device):
    #plot pinn prediction with 2d input
    
    N = 2500
    sN = int(np.sqrt(N))
    
    
    t_plot = torch.zeros((N, 2)).to(device)
    t1 = torch.linspace(tmin[0], tmax[0], sN)
    t2 = torch.linspace(tmin[1], tmax[1], sN)
    buf1, buf2 = torch.meshgrid(t1, t2)
    t_plot[:,0], t_plot[:,1] = buf1.flatten(), buf2.flatten()
    x_plot = fx(t_plot).detach().cpu().numpy().squeeze()
    
    t_plot_test = torch.zeros((N, 2)).to(device)
    t1 = torch.linspace(tmax[0], tmax_coll[0], sN)
    t2 = torch.linspace(tmax[1], tmax_coll[1], sN)
    buf1_test, buf2_test = torch.meshgrid(t1, t2)
    t_plot_test[:,0], t_plot_test[:,1] = buf1_test.flatten(), buf2_test.flatten()
    x_plot_test = fx(t_plot_test).detach().cpu().numpy().squeeze()
    
    
    xnp_ges = []
    xnp_ges_test = []
    for netp in net_pinn:
        netp.eval().to(device)
        xnp_ges.append(netp(t_plot.unsqueeze(1)).detach().cpu().numpy().squeeze())
        xnp_ges_test.append(netp(t_plot_test.unsqueeze(1)).detach().cpu().numpy().squeeze())
        netp.to('cpu')
        
    
    mycmap = plt.get_cmap('gist_earth')
    mycmap2 = plt.get_cmap('autumn')
    fig, ax = plt.subplots(2, np.maximum(2,len(net_pinn)), figsize=(12,8), subplot_kw={'projection': '3d'})
    for i in range(2):
        for j in range(len(net_pinn)):
            if i == 0:
                xnp_buf = xnp_ges
            elif i == 1:
                xnp_buf = xnp_ges_test                
            
            if i == 0:
                surf = ax[i,j].plot_surface(buf1.numpy(), buf2.numpy(), xnp_buf[j].reshape(sN,sN), cmap=mycmap)
                surf = ax[i,j].plot_surface(buf1.numpy(), buf2.numpy(), x_plot.reshape(sN,sN), cmap=mycmap2)
            if i == 1:
                surf = ax[i,j].plot_surface(buf1_test.numpy(), buf2_test.numpy(), xnp_buf[j].reshape(sN,sN), cmap=mycmap)
                surf = ax[i,j].plot_surface(buf1_test.numpy(), buf2_test.numpy(), x_plot_test.reshape(sN,sN), cmap=mycmap2)
            
    plt.show()
    
    
def get_yscale(yscale_opt, ax = -1, jax = 0):
    # get correct axis scale
    
    if type(ax) == int:
        if yscale_opt == 0:
            fplt = plt.plot
        else:
            fplt = plt.semilogy
    else:
        if yscale_opt == 0:
            fplt = ax[jax].plot
        else:
            fplt = ax[jax].semilogy
    return fplt
        
        
def plot_dpar_loss(dparges, dpar_true, lossdges, lossfges, ld_fac, lf_fac, Jges, path):
    #plot pde parameters and losses
    
    Npar = len(dparges)    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))    
    fplt = get_yscale(1, ax, 0)
    
    for jp in range(Npar):
        fplt(np.array(dparges[jp][:]), label='p'+str(jp))
        ax[0].hlines(dpar_true[jp],0,len(dparges[jp]),color='gray',linestyle='--')    
    ax[0].legend()
    
    ax[1].semilogy(np.array(Jges), label='Jges')
    ax[1].semilogy(np.array(lossdges), label='ld (' + str(ld_fac)+')')
    ax[1].semilogy(np.array(lossfges), label='lf (' + str(lf_fac)+')')
    ax[1].legend()
    
    plt.savefig(path + 'dpar.pdf')    
    plt.show()
    
def plot_dpar_statistics(dpargesges, dpar_true, Jgesges, lossdgesges, lossfgesges, Npar, path, step=20):
    #plot statistics of learned pde parameters and losses
    
    Npar = dpargesges[0].shape[0]
    Nmodel = 3
    
    std_list = []
    mu_list = []
    muJ, stdJ, muld, stdld, mulf, stdlf = [[] for j in range(6)]
    
    mu_lists = [mu_list, muJ, muld, mulf]
    std_lists = [std_list, stdJ, stdld, stdlf]
    gesges_lists = [dpargesges, Jgesges, lossdgesges, lossfgesges]
    for j in range(Nmodel):
        for k in range(4):
            i = 1 if k == 0 else 0
            if k == 0:
                mu_lists[k].append((gesges_lists[k][j]).mean(i)[:,::step])
                std_lists[k].append((gesges_lists[k][j]).std(i)[:,::step])
            else:
                mu_lists[k].append((gesges_lists[k][j]).mean(i)[::step])
                std_lists[k].append((gesges_lists[k][j]).std(i)[::step])

    
    cvec = ['r','b','g','c']    
    lvec = ['pinn0', 'pinn-ebm', 'pinn-off', 'pinn0-ebm', 'true']
    kvec = ['dpar','J','ld','lf']
    
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    fplt = get_yscale(0, ax, 0)
    for jp in range(Npar):
        for j in range(Nmodel):
            fplt(np.array(range(0,mu_lists[0][j].shape[1]))*step, mu_lists[0][j][jp], label=lvec[j], linewidth=2, color=cvec[j])
            ax[0].fill_between(np.array(range(0,mu_lists[0][j].shape[1]))*step,mu_lists[0][j][jp] + std_lists[0][j][jp] , mu_lists[0][j][jp] - std_lists[0][j][jp], color=cvec[j], alpha=0.5)    
            
    fplt = get_yscale(1, ax, 1)
    for j in range(Nmodel):
        for k in range(1,4): #should not be inside Npar loop!
            fplt(np.array(range(0,mu_lists[k][j].shape[0]))*step, mu_lists[k][j], label=kvec[k]+lvec[j], linewidth=2, color=cvec[j])
            ax[1].fill_between(np.array(range(0,mu_lists[k][j].shape[0]))*step,mu_lists[k][j] + std_lists[k][j] , mu_lists[k][j] - std_lists[k][j], color=cvec[j], alpha=0.5)    
        ax[0].hlines(dpar_true[jp],0,mu_list[0].shape[1]*step,color='gray',linestyle='--')
        
    #exp
    #ax[0].set_ylim(0.1,0.45)
    #ax[1].set_ylim(0,10)
    #Bessel
    #ax[0].set_ylim(0.4,0.8)
    #ax[1].set_ylim(0,10)
    #NS
    #ax[0].set_ylim(0.0075,0.05)
    
    ax[0].legend()
    ax[1].legend()
    print(path)
    plt.savefig(path + 'dpar_stats.pdf')    
    plt.show()
    
def plot_logL(jm, logLG_ges, logLebm_ges, rmse_ges, path, itest=100):
    #plot logL and RMSE for training and test data
    
    N = len(logLebm_ges[0])
    Ndp = len(logLG_ges[2][0])
    jvec = np.linspace(0,N-1,N)*itest
    
    buf = np.array(range(0,Ndp))*itest
    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    for i in range(2):
        for j in range(2):
            if i == 0:
                if len(logLG_ges[0][j]) > 0:
                    ax[i,j].plot(buf, np.array(logLG_ges[0][j]), color='r', label='pinn0')
                if len(logLebm_ges[1][j]) > 0:
                    ax[i,j].plot(buf, np.array(logLebm_ges[1][j]), color='b', label='pinn ebm')
                if len(logLG_ges[2][j]) > 0:
                    ax[i,j].plot(buf, np.array(logLG_ges[2][j]), color='g', label='pinn off')
                if len(logLG_ges[3][j]) > 0:
                    ax[i,j].plot(buf, np.array(logLG_ges[3][j]), color='c', label='pinn0 ebm')
                #ax[i,j].legend(['pinn0', 'pinn ebm', 'pinn off'])
                if j == 0:
                    ax[i,j].set_title('log L - train')
                else:
                    ax[i,j].set_title('log L - test')
            elif i == 1:
                if len(rmse_ges[0][j]) > 0:
                    ax[i,j].plot(buf, np.array(rmse_ges[0][j]), color='r', label='pinn0')
                if len(rmse_ges[1][j]) > 0:
                    ax[i,j].plot(buf, np.array(rmse_ges[1][j]), color='b', label='pinn ebm')
                if len(rmse_ges[2][j]) > 0:
                    ax[i,j].plot(buf, np.array(rmse_ges[2][j]), color='g', label='pinn off')
                if len(rmse_ges[3][j]) > 0:
                    ax[i,j].plot(buf, np.array(rmse_ges[3][j]), color='c', label='pinn0 ebm')
                #ax[i,j].legend(['pinn0', 'pinn ebm', 'pinn off'])
                if j == 0:
                    ax[i,j].set_title('rmse - train')
                else:
                    ax[i,j].set_title('rmse - test')
    
    plt.savefig(path + 'error.pdf')    
    plt.show()
                    
                    
def plot_error_statistics(logLG_gesges, logLebm_gesges, rmse_gesges, path, itest = 100):
    #plot statistics of logL and RMSE
    
    N = logLG_gesges.shape[3]
    Nrun = logLG_gesges.shape[2]
    
    logLG_mu = logLG_gesges.mean(2)
    logLG_std = logLG_gesges.std(2)
    
    logLebm_mu = logLebm_gesges.mean(2)
    logLebm_std = logLebm_gesges.std(2)
    
    rmse_mu = rmse_gesges.mean(2)
    rmse_std = rmse_gesges.std(2)
    
    cvec = ['r','b','g','c']
    lvec = ['pinn0', 'pinn-ebm', 'pinn-off', 'pinn0-ebm']
    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    for i in range(2):
        for j in range(2):
            for jm in range(3):
                if i == 0:
                    if jm in [1,3]:
                        mu = logLebm_mu[jm,j,:]
                        std = logLebm_std[jm,j,:]
                    else:
                        mu = logLG_mu[jm,j,:]
                        std = logLG_std[jm,j,:]
                elif i == 1:
                    mu = rmse_mu[jm,j,:]
                    std = rmse_std[jm,j,:]
                    
                ax[i,j].plot(np.array(range(0,mu.shape[0]))*itest,mu, label=lvec[jm], linewidth=2, color=cvec[jm])
                ax[i,j].fill_between(np.array(range(0,mu.shape[0]))*itest,mu + std, mu - std, color=cvec[jm], alpha=0.5)
                
            ax[i,j].legend()
            if i == 0:
                if j == 0:
                    ax[i,j].set_title('log L - train')
                else:
                    ax[i,j].set_title('log L - test')
            elif i == 1:
                if j == 0:
                    ax[i,j].set_title('rmse - train')
                else:
                    ax[i,j].set_title('rmse - test')
    
    #exp
    #ax[0,0].set_ylim(-3.5, -2.3)
    #ax[0,1].set_ylim(-15, 4)
    #ax[1,0].set_ylim(0,5)
    #ax[1,1].set_ylim(0,18)
    #ns
    #ax[1,1].set_ylim(0.04,0.35)
    #ax[0,1].set_ylim(-0.9,0.25)
    
    
    plt.savefig(path + 'error_stats.pdf')    
    plt.show()
    return -1
    
                    

def loss_plot(loss_list, loss_legend): 
    #generic loss plot
    
    plt.figure()
    for i in range(len(loss_list)):
        plt.semilogy(loss_list[i])
    plt.legend(loss_legend)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time as tm
import parameters as par
from scipy.optimize import curve_fit

# define a function to fit to for the rates
# P_data is [integral of PL, integral of PR]
def rate_fit_func(P_data,kf,kb):
    return kf*P_data[0,:] - kb*P_data[1,:]

indexToPullRate = -1 # usually -1

ωcs = np.insert(np.linspace(600,1600,100),0,0)*par.cmtoau # wavenumber to au (include 0 to get the rate outside the cavity)
k_vals = np.zeros([len(ωcs)-1,1]) # rate constants
k0 = 0 # create a variable for the rate outside the cavity
for ii in range(len(ωcs)):

    ωc = ωcs[ii] # set current resonant frequency of the cavity
    # ===================================
    # LOADING OF DATA
    cpus = par.Cpus
    Nst = par.nData
    time = par.Sim_time[::par.nskip]
    print(Nst, 'Number of steps')
    ρ = np.zeros((Nst, par.nDW), dtype=np.complex128)
    test = np.zeros((Nst,2), dtype=np.complex128)
    for k in range(cpus):
        # print(ωc/par.cmtoau)
        print(f'./data/rho_{k}_{ωc/par.cmtoau}.txt')
        if ωc == 0:
            ρ += np.loadtxt(f'./data/rho_{k}.txt') # pull from the data outside the cavity
        else:
            # print(k)
            ρ += np.loadtxt(f'./data/rho_{k}_{ωc/par.cmtoau}.txt')
        # print(np.isfinite(ρ))
    ρ /= cpus # averages over all trajectories

    # ===================================
    color = ['#3498db', '#e74c3c' ,'#1abc9c', '#9b59b6', '#e67e22', '#34495e']
    # ========================================
    fig, ax = plt.subplots(figsize = (4.5,4.5))
    tot = np.sum(ρ, axis = 1)
    ax.plot(time[:indexToPullRate]/par.fstoau/1000, ρ[:indexToPullRate,0],  ls = '-', lw = 3, color = color[0],    label = r'$|\nu_L⟩$', alpha = 0.8) 
    ax.plot(time[:indexToPullRate]/par.fstoau/1000, ρ[:indexToPullRate,1],  ls = '-', lw = 3, color = color[1],    label = r'$|\nu_R⟩$', alpha = 0.8) 
    ax.plot(time[:indexToPullRate]/par.fstoau/1000,ρ[:indexToPullRate,2],  ls = '-', lw = 3, color = color[2],   label = r"$|\nu'_L⟩$", alpha = 0.8) 
    ax.plot(time[:indexToPullRate]/par.fstoau/1000,ρ[:indexToPullRate,3],  ls = '-', lw = 3, color = color[3],   label = r"$|\nu'_R⟩$", alpha = 0.8) 
    ax.plot(time[:indexToPullRate]/par.fstoau/1000,tot[:indexToPullRate],      ls = '--',lw = 2, color = color[-1],  label = r"$Tot. Pop$", alpha = 0.8) 

    # state_0 = np.loadtxt('./Deping_data/state_0.txt')
    # state_1 = np.loadtxt('./Deping_data/state_1.txt')
    # state_2 = np.loadtxt('./Deping_data/state_2.txt')
    # state_3 = np.loadtxt('./Deping_data/state_3.txt')

    # ax.plot(state_0[:,0], state_0[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
    # ax.plot(state_1[:,0], state_1[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
    # ax.plot(state_2[:,0], state_2[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
    # ax.plot(state_3[:,0], state_3[:,1], ls = '-.', lw = 1, color = 'black', alpha = 0.9)
    # ax.axhline(0, ls = '--', lw = 1, color = 'black')
    # ax.axhline(1, ls = '--', lw = 1, color = 'black')
    ax.set_xlim(time[0]/par.fstoau/1000,time[indexToPullRate]/par.fstoau/1000)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
    ax.tick_params(which='minor', length=5, direction = 'in')
    ax.set_xlabel('Time (ps)', fontsize = 20)
    ax.set_ylabel('Population', fontsize = 20)
    # ax.set_ylim(0,0.003)


    ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)

    plt.savefig(f'images/popwithcav_{ωc/par.cmtoau}.png', dpi = 300, bbox_inches='tight')
    plt.close()

    # calculate reaction rate at each timestep - really we just need one timestep t = 7ps
    PL = ρ[:,0]
    PR = ρ[:,1]
    dt = (time[1]-time[0])/par.fstoau/1000
    P_data = dt* np.array([np.cumsum(PL),np.cumsum(PR)]) # integral over PL and PR to time t'
    kf = np.ones([Nst-1,1])
    kb = np.ones([Nst-1,1])
    for tp in range(Nst-1):
        params, covariances = curve_fit(rate_fit_func, P_data[:,:(tp+2)], PR[:(tp+2)]) # fit to the rate equation
        kf[tp] = params[0]
        kb[tp] = params[1]
    
    # save rate constant
    if ωc == 0:
        k0 = kf[indexToPullRate]
    else:
        k_vals[ii-1] = kf[indexToPullRate]

    # calculate their reaction rate
    # P0 = state_0[:,1]
    # P1 = state_1[:,1]
    # dt = state_0[1,0] - state_0[0,0]
    # print(state_0[:,0])
    # print(state_1[:,0])
    # P_data = dt* np.vstack((np.cumsum(P0),np.cumsum(P1))) # integral over PL and PR to time t'
    # kf_deping = np.ones([len(P0)-1,1])
    # kb_deping = np.ones([len(P0)-1,1])
    # print(P_data)
    # for tp in range(len(P0)-1):
    #     print(P_data[:,:(tp+2)])
    #     print(P1[:(tp+2)])
    #     params, covariances = curve_fit(rate_fit_func, P_data[:,:(tp+2)], P1[:(tp+2)]) # fit to the rate equation
    #     kf_deping[tp] = params[0]
    #     kb_deping[tp] = params[1]
    # print(kf,kb)

    # plot reaction rate - the forward rate is the one we call the reaction rate
    fig, ax = plt.subplots(2,figsize = (4.5,4.5))
    if indexToPullRate == -1:
        ax[0].plot(time[1:]/par.fstoau/1000, kf[:],  ls = '-', lw = 3, color = color[0],    label = 'kf', alpha = 0.8)
        ax[1].plot(time[1:]/par.fstoau/1000, kb[:],  ls = '-', lw = 3, color = color[0],    label = 'kb', alpha = 0.8)
    else:
        ax[0].plot(time[1:(indexToPullRate+1)]/par.fstoau/1000, kf[:indexToPullRate],  ls = '-', lw = 3, color = color[0],    label = 'kf', alpha = 0.8)
        ax[1].plot(time[1:(indexToPullRate+1)]/par.fstoau/1000, kb[:indexToPullRate],  ls = '-', lw = 3, color = color[0],    label = 'kb', alpha = 0.8)
    # ax[0].plot(state_0[1:,0], kf_deping, ls = '-.', lw = 1, color = 'black', alpha = 0.9, label = 'kf Deping Data')
    # ax[1].plot(state_1[1:,0], kb_deping, ls = '-.', lw = 1, color = 'black', alpha = 0.9, label = 'kb Deping Data')
    # ax[0].set_xlabel('Time (fs)', fontsize = 20)
    # ax[0].set_ylabel('Reaction Rate', fontsize = 20)
    # ax[0].set_xlim(7,time[-1]/par.fstoau/1000)
    ax[1].set_xlabel('Time (ps)', fontsize = 20)
    ax[1].set_ylabel('Reaction Rate (1/ps)', fontsize = 20)
    # ax[1].set_xlim(7,time[-1]/par.fstoau/1000)
    # ax[1].set_ylim(-3,3)
    ax[0].legend()
    ax[1].legend()
    plt.savefig(f'images/ratewithcav_{ωc/par.cmtoau}.png', dpi = 300, bbox_inches='tight')
    plt.close()

# plot the rate constant k/k0
fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(ωcs[1:]/par.cmtoau, k_vals/k0, lw = 3, color = color[0], label = r'$\eta_c = 1.25e-3$', alpha = 0.8)
ax.set_xlabel('ωc (1/cm)', fontsize = 20)
ax.set_ylabel('k/k0', fontsize = 20)
ax.legend()
plt.savefig(f'images/relativeRates.png', dpi = 300, bbox_inches='tight')
plt.close()

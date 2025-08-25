import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time as tm
import parameters as par
from scipy.optimize import curve_fit
import pickle

# define a function to fit to for the rates
# P_data is [integral of PL, integral of PR]
def rate_fit_func(P_data,kf,kb):
    return kf*P_data[0,:] - kb*P_data[1,:]

indexToPullRate = 4725#-1 # usually -1
loadDataFlag = 1 # flag to decide whether population data needs to be loaded

ωcs = np.array([0,800,1150,1190,1600])*par.cmtoau#np.array([0,800,1000,1130,1150,1170,1180,1190,1200,1210,1230,1250,1400,1600])*par.cmtoau#np.insert(np.linspace(700,1700,101),0,0)*par.cmtoau # wavenumber to au (include 0 to get the rate outside the cavity)
k_vals = np.zeros([len(ωcs)-1,1]) # rate constants
k0 = 0 # create a variable for the rate outside the cavity

# ===================================
color = ['#3498db', '#e74c3c' ,'#1abc9c', '#9b59b6', '#e67e22', '#34495e']
# =======================================

if loadDataFlag:
    for ii in range(len(ωcs)):

        ωc = ωcs[ii] # set current resonant frequency of the cavity
        # ===================================
        # LOADING OF DATA
        cpus = par.Cpus
        Nst = par.nData
        time = par.Sim_time[::par.nskip]
        print(Nst, 'Number of steps')
        ρ = np.zeros((Nst, par.hamDim), dtype=np.float64)
        # ρ = np.zeros((Nst, par.nDW), dtype=np.complex128)
        # test = np.zeros((Nst,2), dtype=np.complex128)
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

        ρR = np.zeros((par.nData,par.nDW))
        for k in range(par.nDW):
            ρR[:,k] = np.sum(ρ[:,par.nsolvent*par.nSlevels*par.nPhLevels*k:par.nsolvent*par.nSlevels*par.nPhLevels*(k+1)],axis=1)

        # plot populations
        fig, ax = plt.subplots(figsize = (4.5,4.5))
        tot = np.sum(ρR, axis = 1)
        ax.plot(time/par.fstoau/1000, ρR[:,0],  ls = '-', lw = 3, color = color[0],    label = r'$|\nu_L⟩$', alpha = 0.8) 
        ax.plot(time/par.fstoau/1000, ρR[:,1],  ls = '-', lw = 3, color = color[1],    label = r'$|\nu_R⟩$', alpha = 0.8) 
        ax.plot(time/par.fstoau/1000,ρR[:,2],  ls = '-', lw = 3, color = color[2],   label = r"$|\nu'_L⟩$", alpha = 0.8) 
        ax.plot(time/par.fstoau/1000,ρR[:,3],  ls = '-', lw = 3, color = color[3],   label = r"$|\nu'_R⟩$", alpha = 0.8) 
        if len(ρR[0,:]) > 4:
            ax.plot(time/par.fstoau/1000,ρR[:,4],  ls = '-', lw = 3, color = color[4],   label = r"$|\nu_4⟩$", alpha = 0.8) 
        ax.plot(time/par.fstoau/1000,tot,      ls = '--',lw = 2, color = color[-1],  label = r"$Tot. Pop$", alpha = 0.8)
        ax.axhline(0, ls = '--', lw = 1, color = 'black')
        ax.axhline(1, ls = '--', lw = 1, color = 'black')
        ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
        # ax.set_ylim(0,0.00005)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
        ax.tick_params(which='minor', length=5, direction = 'in')
        ax.set_xlabel('Time (ps)', fontsize = 20)
        ax.set_ylabel('Population', fontsize = 20)
        ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)
        plt.savefig(f'images/popwithcav_{ωc/par.cmtoau}.png', dpi = 300, bbox_inches='tight')
        plt.close()

        # calculate reaction rate at each timestep - really we just need one timestep t = 7ps
        PL = ρR[:,0]
        PR = ρR[:,1]
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
            print("k0: ", kf[-10:])
        else:
            k_vals[ii-1] = kf[indexToPullRate]

        # plot population of solvent
        ρQ = np.zeros((par.nData,par.nSlevels))
        for k in range(par.nSlevels):
            ρQ[:,k] = np.sum(ρ[:,k::2],axis=1)

        # print(ρQ)

        # plot populations
        fig, ax = plt.subplots(figsize = (4.5,4.5))
        tot = np.sum(ρQ, axis = 1)
        ax.plot(time/par.fstoau/1000, ρQ[:,0],  ls = '-', lw = 3, color = color[0],    label = r'$|\nu_L⟩$', alpha = 0.8) 
        ax.plot(time/par.fstoau/1000, ρQ[:,1],  ls = '-', lw = 3, color = color[1],    label = r"$|\nu'_L⟩$", alpha = 0.8) 
        ax.plot(time/par.fstoau/1000,tot,      ls = '--',lw = 2, color = color[-1],  label = r"$Tot. Pop$", alpha = 0.8)
        ax.axhline(0, ls = '--', lw = 1, color = 'black')
        ax.axhline(1, ls = '--', lw = 1, color = 'black')
        ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
        # ax.set_ylim(0,0.00005)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
        ax.tick_params(which='minor', length=5, direction = 'in')
        ax.set_xlabel('Time (ps)', fontsize = 20)
        ax.set_ylabel('Population', fontsize = 20)
        ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)
        plt.savefig(f'images/popwithcav_{ωc/par.cmtoau}_solvent.png', dpi = 300, bbox_inches='tight')
        plt.close()

        # plot population of cavity
        ρc = np.zeros((par.nData,par.nPhLevels))
        for k in range(par.nSlevels):
            ρc[:,k] = np.sum(ρ[:,(2*k)::4],axis=1) + np.sum(ρ[:,(2*k+1)::4],axis=1)

        # print(ρc)

        # plot populations
        fig, ax = plt.subplots(figsize = (4.5,4.5))
        tot = np.sum(ρc, axis = 1)
        ax.plot(time/par.fstoau/1000, ρc[:,0],  ls = '-', lw = 3, color = color[0],    label = r'$|\nu_L⟩$', alpha = 0.8) 
        ax.plot(time/par.fstoau/1000, ρc[:,1],  ls = '-', lw = 3, color = color[1],    label = r"$|\nu'_L⟩$", alpha = 0.8) 
        ax.plot(time/par.fstoau/1000,tot,      ls = '--',lw = 2, color = color[-1],  label = r"$Tot. Pop$", alpha = 0.8)
        ax.axhline(0, ls = '--', lw = 1, color = 'black')
        ax.axhline(1, ls = '--', lw = 1, color = 'black')
        ax.set_xlim(time[0]/par.fstoau/1000,time[-1]/par.fstoau/1000)
        # ax.set_ylim(0,0.00005)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='major', length=12, labelsize = 13, direction = 'in')
        ax.tick_params(which='minor', length=5, direction = 'in')
        ax.set_xlabel('Time (ps)', fontsize = 20)
        ax.set_ylabel('Population', fontsize = 20)
        ax.legend(title ='Vibrational States', loc=0, frameon = False, fontsize = 9, handlelength=1, title_fontsize = 9, labelspacing = 0.2)
        plt.savefig(f'images/popwithcav_{ωc/par.cmtoau}_cavity.png', dpi = 300, bbox_inches='tight')
        plt.close()


# pull rate data from Sebastian's paper (https://pubs.acs.org/doi/10.1021/jacs.5c03182)
# k_5 = np.loadtxt('./data/k_wc_scan_etac_0.005.txt')

# pull data from previous runs
# if ~loadDataFlag:
#     with open('data/RelativeRates_Rabi114.txt','rb') as f:
#         ωcs114,krel114 = pickle.load(f)


# plot the rate constant k/k0
fig, ax = plt.subplots(figsize = (4.5,4.5))
# ax.plot(k_5[:,0],k_5[:,1]/9.077e-08, c = color[3], linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '6',label='HEOM')
# ax.plot(ωcs114[1:]/par.cmtoau, krel114, lw = 3, color = color[3], label = r'$\Omega_R = 114$', alpha = 0.8)
if loadDataFlag:
    ax.plot(ωcs[1:]/par.cmtoau, k_vals/k0, lw = 3, color = color[1], label = '$\Omega_R_qs = 114$', alpha = 0.8)
    print(k_vals/k0)
else:
    # ax.plot(ωcs114[1:]/par.cmtoau, krel114, lw = 3, color = color[3], label = r'$\Omega_R = 114$', alpha = 0.8)
    pass
ax.set_xlabel('ωc (1/cm)', fontsize = 20)
ax.set_ylabel('k/k0', fontsize = 20)
ax.legend()
try:
    plt.savefig('./images/relativeRates.png', dpi = 300, bbox_inches='tight')
except:
    plt.savefig('images/relativeRates.png', dpi = 300, bbox_inches='tight')
plt.close()

# save data to not need to load it next time
if loadDataFlag:
    with open(f'data/RelativeRates_Rabi{par.Ω}_qsc.txt','wb') as f:
        pickle.dump([ωcs,k_vals/k0],f)
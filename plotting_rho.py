import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time as tm
import parameters as par
from scipy.optimize import curve_fit
import pickle
import os.path

# define a function to fit to for the rates
# P_data is [integral of PL, integral of PR]
def rate_fit_func(P_data,kf,kb):
    return kf*P_data[0,:] - kb*P_data[1,:]

indexToPullRate = -1 # usually -1
loadDataFlag = 0 # flag to decide whether population data needs to be loaded

ωcs = np.insert(np.linspace(700,1700,101),0,0)*par.cmtoau # wavenumber to au (include 0 to get the rate outside the cavity)
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
        ρ = np.zeros((Nst, par.nDW), dtype=np.complex128)
        test = np.zeros((Nst,2), dtype=np.complex128)
        for k in range(cpus):
            print(f'./data/rho_{k}_{ωc/par.cmtoau}.txt')
            if ωc == 0:
                ρ += np.loadtxt(f'./data/rho_{k}.txt') # pull from the data outside the cavity
            else:
                ρ += np.loadtxt(f'./data/rho_{k}_{ωc/par.cmtoau}.txt')
        ρ /= cpus # averages over all trajectories

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

# pull rate data from Sebastian's paper (https://pubs.acs.org/doi/10.1021/jacs.5c03182)
k_5 = np.loadtxt('./data/k_wc_scan_etac_0.005.txt')
k_25 = np.loadtxt('./data/k_wc_scan_etac_0.0025.txt')

# pull data from previous runs
with open('data/RelativeRates_Rabi114.txt','rb') as f:
    ωcs114,krel114 = pickle.load(f)
with open('data/RelativeRates_Rabi57.txt','rb') as f:
    ωcs57,krel57 = pickle.load(f)


# plot the rate constant k/k0
fig, ax = plt.subplots(figsize = (4.5,4.5))
ax.plot(ωcs57[1:]/par.cmtoau, krel57, lw = 3, color = color[1], label = r'$\Omega_R = 57$', alpha = 0.8)
ax.plot(ωcs114[1:]/par.cmtoau, krel114, lw = 3, color = color[2], label = r'$\Omega_R = 114$', alpha = 0.8)
if loadDataFlag:
    ax.plot(ωcs[1:]/par.cmtoau, k_vals/k0, lw = 3, color = color[1], label = rf'$\Omega_R = {par.Ω}$', alpha = 0.8)
# else:
#     ax.plot(ωcs114[1:]/par.cmtoau, krel114, lw = 3, color = color[3], label = r'$\Omega_R = 114$', alpha = 0.8)
ax.plot(k_25[:,0],k_25[:,1]/9.077e-08, c = color[1], linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '6', label = 'HEOM 57')
ax.plot(k_5[:,0],k_5[:,1]/9.077e-08, c = color[2], linestyle = ' ', marker = 'o', fillstyle= 'full', markersize = '6', label = 'HEOM 114')
ax.set_xlabel('ωc (1/cm)', fontsize = 20)
ax.set_ylabel('k/k0', fontsize = 20)
ax.legend()
plt.savefig(f'images/relativeRates.png', dpi = 300, bbox_inches='tight')
plt.close()

# save data to not need to load it next time
if loadDataFlag:
    if os.path.isfile(f'data/RelativeRates_Rabi{par.Ω}.txt'): # try not to overwrite important data that took a whole day to get
        overwriteFlag = input(f'Do you really want to overwrite the file data/RelativeRates_Rabi{par.Ω}.txt? (y/n) ')
        if overwriteFlag == 'y':
            with open(f'data/RelativeRates_Rabi{par.Ω}.txt','wb') as f:
                pickle.dump([ωcs,k_vals/k0],f)
        else:
            print("The file wasn't overwritten")
    else:
        with open(f'data/RelativeRates_Rabi{par.Ω}.txt','wb') as f:
            pickle.dump([ωcs,k_vals/k0],f)

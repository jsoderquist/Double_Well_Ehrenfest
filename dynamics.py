#!/software/anaconda3/2020.11/bin/python
#SBATCH -p standard
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qjob.out
#SBATCH --error=qjob.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import time as tm
import sys, os
import MFE as method
import parameters as par
import TrajClass as tc
import model
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelmax
# =================================

ωcs = np.insert(np.linspace(700,1700,101),0,1189.7)*par.cmtoau#np.array([1189.7])*par.cmtoau # wavenumber to au
for ωc in ωcs:

    # =========================
    # Parallelization
    # =========================
    # RUN PARALLEL TRAJECTORIES
    # THE NUMBER OF TRAJECTORIES PER JOB (j) IS DETERMINED BASED ON THE NUMBER OF CPUS (par.Cpus) AND TOTAL TRAJECTORIES (par.NTraj)
    # j = NTraj / Cpus
    parallel = par.parallel

    if (parallel == True):
        sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # INCLUDE PARENT DIRECTORY WHICH HAS METHOD AND MODEL FILES
        JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"])               # GET ID OF THIS JOB
        TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"])             # GET ID OF THIS TASK WITHIN THE ARRAY 

        nrank = int(TASKID)                                         # JOD ID FOR A JOB 
        size  = par.Cpus                                            # TOTAL NUMBER OF PROCESSOR AVAILABLE
    else:
        nrank = 0
        size  = 1

    # =================================
    # COMPILATION 
    # =================================
    # WITH JIT, THE CODE MUST BE COMPILE FIRST. RUN THE CODE FOR ONLY TWO TIME STEPS FIRST
    # com_ti = tm.time()
    # nDW_dummy = par.nDW
    # ndof_dummy = par.ndof
    # nsteps_dummy = 2
    # data_dummy = tc.trajData(nDW_dummy, ndof_dummy, nsteps_dummy, nsteps_dummy)
    # data_dummy.ρt = par.ρ0

    # # MODEL FUNCTIONS =================
    # model.initR(data_dummy)
    # model.H_BC(data_dummy)

    # # METHOD FUNCTIONS ================
    # method.Force1(data_dummy)
    # method.RK4(data_dummy)
    # method.VelVer(data_dummy)
    # method.run_traj(data_dummy)

    # com_tf = tm.time()
    # print(f'Compilation time --> {np.round(com_tf - com_ti,2)} s or {np.round((com_tf - com_ti)/60,2)} min')


    # =================================
    # SIMULATION
    # =================================
    # DIVIDE THE NUMBER OF TRAJECTORIES PER JOB BASE ON THE NUMBER OF PROCESSORS AND TOTAL TRAJECTORIES
    tot_Tasks = par.NTraj
    NTasks = tot_Tasks//size
    NRem = tot_Tasks - (NTasks*size)
    TaskArray = [i for i in range(nrank * NTasks , (nrank+1) * NTasks)]
    for i in range(NRem):
        if i == nrank: 
            TaskArray.append((NTasks*size)+i)
    TaskArray = np.array(TaskArray)                                  # CONTAINS THE NUMBER OF TRAJECTORIES ASSIGNED TO EACH JOB
    # =================================

    trajData = tc.trajData(par.nDW, par.ndof, par.NSteps, par.nData, ωc) # INITIATE THE TIME DEPENDENT DATA
    ρw = np.zeros((par.nData, par.nDW))                              # DENSITY MATRIX AVERAGED OVER THE NUMBER OF TRAJECTORIES ASSIGNED TO THIS JOB

    sim_ti = tm.time()
    trajData.cj, trajData.ωj = par.calc_cjωj(ωc) # calculate these here so that we can automate the system
    trajData.dHij = par.dHij_cons(trajData.cj)

    # plot spectral density
    wvals = np.linspace(700,1700,2000) # omega values in spectral density plot
    J = par.J_eff(1/par.τc, par.ηc, ωc, wvals*par.cmtoau,par.λQ,par.γQ,par.nsolvent,1,par.Λ,par.ωQ)
    # J = par.J_DrudeL(par.λD, par.γD, np.linspace(600,1600,2000)*par.cmtoau)
    # for n in range(par.ndof):
    #     plt.axvline(trajData.ωj[n]/par.cmtoau, ls = '-.', color = 'black', lw = 1)
    plt.plot(wvals,J, lw = 3, label=ωc)#, c = 'r')
    if par.ηc == 0: # no cavity case
        try:
            plt.savefig('../images/spectralDen.png')
        except:
            plt.savefig('./images/spectralDen.png')
    else:
        try:
            plt.savefig('../images/spectralDenInCav.png')
        except:
            plt.savefig('./images/spectralDenInCav.png')

    extrema = wvals[argrelmax(J)]
    if len(extrema) > 1:
        checkΩ = extrema[1] - extrema[0]
        print("Ω: ", checkΩ, " μQ: ",checkΩ/2/np.sqrt(par.nsolvent)/par.ηc/(ωc/par.cmtoau))
        print(extrema)
    else:
        print(len(extrema))

    # sys.exit("I only want to plot the spectral density right now")
    
    # trajData.cj = par.calc_cj(ωc)
    for i in range(len(TaskArray)):
        method.run_traj(trajData)
        ρw += trajData.ρw
        # print(trajData.H_bc," ",trajData.dHij)
    sim_tf = tm.time()
    print(f'Simulation time --> {np.round(sim_tf - sim_ti,2)} s or {np.round((sim_tf - sim_ti)/60,2)} min')
    print(' ================================================================================================= ')

    if par.ηc == 0: # no cavity case
        try:
            np.savetxt(f'../data/rho_{nrank}.txt', ρw/len(TaskArray))   # RUN IN PARALLEL
        except:
            np.savetxt(f'./data/rho_{nrank}.txt', ρw/len(TaskArray))    # RUN IN SERIES
    else:
        try:
            np.savetxt(f'../data/rho_{nrank}_{ωc/par.cmtoau}.txt', ρw/len(TaskArray))   # RUN IN PARALLEL
        except:
            np.savetxt(f'./data/rho_{nrank}_{ωc/par.cmtoau}.txt', ρw/len(TaskArray))    # RUN IN SERIES
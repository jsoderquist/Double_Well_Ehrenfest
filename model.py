#!/software/anaconda3/2020.11/bin/python
#SBATCH -p action
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import numba as nb
import parameters as par
# import functools as ft
# ===========================

# ELECTRONIC - BATH COUPLING HAMILTONIAN
@nb.jit(nopython=True, fastmath=True)
def H_BC(data):
    # IR = np.eye(par.nDW) # identity matrix of R
    # IQ = np.eye(par.nSlevels) # identity matrix of Q
    # Hbc  = np.zeros((par.nDW*par.nSlevels**par.nsolvent,par.nDW*par.nSlevels**par.nsolvent), dtype = np.complex128)
    Hbc  = np.zeros((par.nDW,par.nDW), dtype = np.complex128)

    # temp = (np.sum(data.cj[:par.ndofb] * data.x[:par.ndofb]) + np.sum(data.cj[-par.ndofc:] * data.x[-par.ndofc:])) * par.R # multiply in R bath couplings and cavity couplings
    # temp = np.kron(ft.reduce(np.kron,[IQ]*par.nsolvent),temp) # extend the space to the space of all solvents after coupling to R
    # H_BC -= temp # subtract out couplings
    Hbc -= np.sum(data.cj[:] * data.x[:]) * par.R # needs to be R coupled only to the first 300 and then Q coupled to the solvent ones, and the cavity probably couples to both?
    # print(data.x," ",par.R) # it looks like data.x is turning to nan
    # print(Hbc)
    data.H_bc = Hbc * 1.0

# INITIALIZE BATH DOF
@nb.jit(nopython=True, fastmath=True)
def initR(data):
    data.x[:] = 0
    data.P[:] = 0

    β  = par.β
    # ωj = par.calc_ωj(ωc)

    # WIGNER DISTRIBUTION FOR POSITION AND MOMENTA.
    # SAMPLED FROM A GAUSSIAN DISTRIBUTION WITH STANDARD DEVIATION σx AND σP.
    σP = np.sqrt(data.ωj / (2 * np.tanh(0.5*β*data.ωj)))
    σx = σP/data.ωj
    # print("σx: ",σx)
    # print("σP: ",σP)

    data.x[:] = np.random.normal(loc=0.0, scale=1.0, size= len(data.ωj)) * σx
    data.P[:] = np.random.normal(loc=0.0, scale=1.0, size= len(data.ωj)) * σP
    # print("x: ",data.x)


# print('================')
# print(np.real(np.round(par.R,3)))
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
# ===========================

# ELECTRONIC - BATH COUPLING HAMILTONIAN
@nb.jit(nopython=True, fastmath=True)
def H_BC(data):
    Hbc  = np.zeros((par.nDW,par.nDW), dtype = np.complex128)
    # cj = par.calc_cj(ωc)
    Hbc -= np.sum(data.cjDWC[:] * data.x[:]) * par.R 
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
    σP = np.sqrt(data.ωjDWC / (2 * np.tanh(0.5*β*data.ωjDWC)))
    σx = σP/data.ωjDWC

    data.x[:] = np.random.normal(loc=0.0, scale=1.0, size= len(data.ωjDWC)) * σx
    data.P[:] = np.random.normal(loc=0.0, scale=1.0, size= len(data.ωjDWC)) * σP


# print('================')
# print(np.real(np.round(par.R,3)))

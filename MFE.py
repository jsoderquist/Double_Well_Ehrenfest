import numpy as np
import numba as nb
import model
import parameters as par
import sys
# ==================================

# FORCE 1 UPDATE FUNCTION
@nb.jit(nopython=True, fastmath=True)
def Force1(data):
    data.F1[:]    = 0
    data.F1[:]   -= data.ωj[:]**2 * data.x[:] 
    par_sum  = np.sum(data.dHij * data.ρt * 1.0, axis= 1).real    # NUMBA DOES NOT ALLOW THE SUMMATION OVER TWO AXIS, MUST BE DONE IN TWO STEPS
    data.F1[:]   -= np.sum(par_sum, axis = 1).real

# FORCE 2 UPDATE FUNCTION
@nb.jit(nopython=True, fastmath=True)
def Force2(data):
    data.F2[:]    = 0
    data.F2[:]   -= data.ωj[:]**2 * data.x[:] 
    par_sum = np.sum(data.dHij * data.ρt * 1.0, axis= 1).real    # NUMBA DOES NOT ALLOW THE SUMMATION OVER TWO AXIS, MUST BE DONE IN TWO STEPS
    data.F2[:]   -= np.sum(par_sum, axis = 1).real

# RUNGE - KUTTA PROPAGATOR
# THIS DOES HALF OF THE ELETRONIC STEPS, par.Estep/2 | Estep MUST BE EVEN!!!
@nb.jit(nopython=True, fastmath=True)
def RK4(data):
    H  = par.Hel_cons(data) * 1.0 + data.H_bc * 1.0
    # if ~np.isfinite(H).all():
    #         raise Exception(f"H is not finite. Hel: {par.Hel_cons(data)} H_bc: {data.H_bc} rho: {data.ρt}")
    # print("Hbc: ",data.H_bc) # H is blowing up, Helcons really is constant
    ρ  = data.ρt * 1.0
    dt = par.dtE * 1.0
    for k in range(int(par.Estep/2)):
        k1 = von_Newman(ρ.copy(), H.copy())
        k2 = von_Newman(ρ.copy() + 0.5 * dt * k1, H.copy())
        k3 = von_Newman(ρ.copy() + 0.5 * dt * k2, H.copy())
        k4 = von_Newman(ρ.copy() + dt * k3, H.copy())
        ρ  = ρ.copy() + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    data.ρt = 1.0 * ρ

# VON - NEWMAN EQUATION
@nb.jit(nopython=True, fastmath=True)
def von_Newman(ρf, H):
    return -1j * (H @ ρf - ρf @ H)

#  VELOCITY VERLET PROPAGATOR
@nb.jit(nopython=True, fastmath=True)
def VelVer(data) : 
    data.v[:] = data.P[:]/par.M * 1.0                                           # VELOCITY 
    # if ~np.isfinite(data.v).all():
    #         raise Exception(f"velocity is not finite. v: {data.v} F2: {data.F2} rho: {data.ρt}")
    # print("P: ",data.P)          
    RK4(data)                                                                   # ELECTRONIC UPDATE | RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!
    # if ~np.isfinite(data.ρt).all():
    #         raise Exception(f"Rho is not finite after Rk41. dHij: {data.dHij} F2: {data.F2} rho: {data.ρt}")
    data.x[:] += data.v[:] * par.dtN + 0.5 * data.F1[:] * par.dtN**2 / par.M    # POSITION UPDATE
    # print("v: ",data.v) # blows up
    # print("F1: ",data.F1) # blows up
    model.H_BC(data)                                                            # ELECTRONIC HAMILTONIAN UPDATE | BATH POSITION DEPEDENT PART (CHECK model.py FILE) 
    RK4(data)                                                                   # ELECTRONIC UPDATE | RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!
    # if ~np.isfinite(data.ρt).all():
    #         raise Exception(f"Rho is not finite after Rk42. dHij: {data.dHij} F2: {data.F2} rho: {data.ρt}")
    Force2(data)                                                                # FORCE AT t2
    # if ~np.isfinite(data.F2).all():
    #         raise Exception(f"Force 2 is not finite. dHij: {data.dHij} F2: {data.F2} rho: {data.ρt}")
    data.v[:] += 0.5 * (data.F1[:] + data.F2[:]) * par.dtN / par.M              # VELOCITY UPDATE
    data.P[:]  = data.v[:] * par.M * 1.0                                        # MOMENTUM UPDATE
    data.F1[:] = data.F2[:] * 1.0                                               # SET F1 AS F2 FOR THE NEXT STEP
    # if ~np.isfinite(data.ρt).all():
    #         raise Exception(f"Rho is not finite in a weird place. dHij: {data.dHij} F2: {data.F2} rho: {data.ρt}")
    # ======================================================

# RUN TRAJECTORIES
@nb.jit(nopython=True, fastmath=True)
def run_traj(data):
    model.initR(data)                           # INITIALIZE x AND P FOR ALL BATH MODES
    data.ρt = par.ρ0                            # INITIAL DENSITY MATRTIX
    # if ~np.isfinite(data.ρt).all():
    #         raise Exception(f"Rho is not finite. rho0: {par.ρ0} rho: {data.ρt}")
    # ωj = par.calc_ωj(ωc)                            # calculate ωj for this set of ωc
    Force1(data)                                # FORCE 1 AT t = 0
    # if ~np.isfinite(data.ρt).all():
    #         raise Exception(f"Rho is not finite after Force1. dHij: {data.dHij} F1: {data.F1} rho: {data.ρt}")
    model.H_BC(data)                            # INITIAL ELECTRONIC HAMILTONIAN UPDATE | BATH POSITION DEPEDENT PART 

    iskip = 0
    for st in range(data.nSteps):
        if (st % par.nskip == 0):               # WRITTING OF THE DENSITY MATRIX | SAVE ONLY THE DIAGONAL ELEMENTS
            ρ = data.ρt
            for k in range(par.nDW):
                data.ρw[iskip,k] = np.real(ρ[k,k])
            iskip += 1

        VelVer(data)                            # EVOLUTION OF THE SYSTEM FOR nsteps

        # if ~np.isfinite(data.ρt).all():
        #     raise Exception(f"Rho is not finite. dHij: {data.dHij} ωj: {data.ωj} H_bc: {data.H_bc} Hel_cons: {par.Hel_cons(data)} dt: {par.dtE} F2: {data.F2} F1: {data.F1} v: {data.v} x: {data.x} rho: {data.ρt}")
        
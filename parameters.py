#!/software/anaconda3/2020.11/bin/python
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qbath.out
#SBATCH --error=qbath.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
# ==================================

# FUNCTIONS
# ==================================
# DOUBLE WELL R IN ENERGY BASIS
@nb.jit(nopython=True, fastmath=True)
def Rx(nDW):
    pos_DW = np.zeros((nDW,nDW), dtype = np.complex128)
    for j in range(nDW):
        for i in range(nDW):
            avg_pos = diaV[:,j].conjugate() * x0 * diaV[:,i]
            pos_DW[j,i] = np.trapz(avg_pos,x0,dx)
    return pos_DW

# solvent Q IN ENERGY BASIS
# @nb.jit(nopython=True, fastmath=True)
# def Qx(nSlevels):
#     pos_solv = np.zeros((nSlevels,nSlevels), dtype = np.complex128)
#     for j in range(nSlevels):
#         for i in range(nSlevels):
#             avg_pos = VS[:,j].conjugate() * xS0 * VS[:,i]
#             pos_solv[j,i] = np.trapz(avg_pos,xS0,dSx)
#     return pos_solv

# DOUBLE WELL POTENTIAL ENERGY
@nb.jit(nopython=True, fastmath=True)
def DW(x,m,wDW):
    # m = 1836 # I don't know why this is being redefined here. It's already given to the function as 1, but it's not used anyway
    Eb = 2250 * cmtoau
    V = -(m*wDW**2 / 2) * x**2 + (m**2*wDW**4 / (16 * Eb)) * x**4
    return V - min(V)

# solvent potential
# @nb.jit(nopython=True, fastmath=True)
# def solvPot(x,m,ωS):
#     m = 1 # I don't know why this is being redefined here. It's already given to the function as 1, but it's not used anyway
#     V = 0.5*m*ωS**2*x**2
#     return V - min(V)

# KINETIC ENERGY
@nb.jit(nopython=True, fastmath=True)
def T(x,m):
    dx = x[0] - x[1]
    N = len(x)
    K = np.pi/dx
    Kin = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Kin[i,j] = K**2/3 * (1 + 2/N**2)
            else:
                Kin[i,j] = 2*K**2/N**2 * (-1)**(j-i)/(np.sin(np.pi * (j-i)/N))**2 
    return 1/(2*m) * Kin

# DISCRETE VARIABLE REPRESENTATION FUNCTION
# diagonalizes the P^2/2M + V part of H as the basis
@nb.jit(nopython=True, fastmath=True)
def DVR(x,m,wDW):
    V = DW(x,m,wDW)
    V = np.diag(V)
    K = T(x,m)
    E, V = np.linalg.eigh(V+K)
    return E, V

# DISCRETE VARIABLE REPRESENTATION FUNCTION for solvent
# diagonalizes the P^2/2M + V part of H as the basis
# @nb.jit(nopython=True, fastmath=True)
# def DVRS(x,m,ωS):
#     V = solvPot(x,m,ωS)
#     V = np.diag(V)
#     K = T(x,m)
#     E, V = np.linalg.eigh(V+K)
#     return E, V

# DRUDE - LORENTZ SPECTRAL DENSITY
@nb.jit(nopython=True, fastmath=True)
def J_DrudeL(λ, γ, ω):
    return (2 * γ * λ * ω) / (ω**2 + γ**2)

# Effective Cavity SPECTRAL DENSITY
@nb.jit(nopython=True, fastmath=True)
def J_eff(α, ηc, ωc, ω,λQ,γQ,nsolvent,χ,Λ,ωQ):
    ΓQ = 2*λQ/γQ + (2*nsolvent*χ**2*ωc**3*ηc**2*α)/((ωc**2 - ω**2)**2 + α**2*ω**2)
    scriptR = (2*nsolvent*χ**2*ωc*ηc**2*ω**2)*(ω**2 - ωc**2 + α**2)/((ωc**2 - ω**2)**2 + α**2*ω**2)
    return (Λ*ωQ**2*ω*ΓQ)/((ωQ**2 - ω**2 + scriptR)**2 + (ω*ΓQ)**2)

# BATH PARAMETERS
@nb.jit(nopython=True, fastmath=True)
def BathParam(λD, γD, N, num):    
    ωj = np.zeros((N))
    cj = np.zeros((N), dtype = np.complex128)

    if num == False:
    # ANALYTIC DISCRETIZATION OF THE DRUDE - LORENTZ SPECTRAL DENSITY 
    # Huo. P., et al (Mol. Phys. 2012, 110, 1035–1052)
        arr = np.arange(0,N,1) + 1
        ω_max = 10 * γD
        ωj[:] = γD * np.tan(arr/N * np.arctan(ω_max/γD))
        cj[:] = 2 * ωj[:] * np.sqrt(λD * np.arctan(ω_max/γD)/(np.pi * N))
    
    else:
        ω  = np.linspace(1E-10,100*γD,50000)                   # FREQUENCY SCAN FOR BATH FREQUENCIES
        dω = ω[1] - ω[0]
    # NUMERICAL DISCRETIZATION OF SPECTRAL DENSITY
    # Walters, P. L.; et al.  J Comput Chem 2017, 38 (2), 110–115. https://doi.org/10.1002/jcc.24527.

        J = J_DrudeL(λD, γD, ω)
    
        Fω = np.zeros(len(ω)) # LHS of eq 2.6?
        for i in range(len(ω)):
            Fω[i] = (4/np.pi) * np.sum(J[:i]/ω[:i]) * dω

        λs =  Fω[-1] # equation 2.5
        for i in range(N):
            costfunc = np.abs(Fω-(((float(i)-0.5)/float(N))*λs)) # using equation 2.6 - I corrected the minus sign (sebastian had i+0.5)
            m = np.argmin((costfunc))
            ωj[i] = ω[m]
        mj = M # let all molecules have equal mass
        cj[:] = ωj[:] * np.sqrt(mj) * ((λs/(2*float(N)))**0.5) # see paragraph under eq 2.6 and earlier paragraph that says cj = κ*sqrt(m_j)*ω_j
    return cj, ωj

# CAVITY BATH PARAMETERS
@nb.jit(nopython=True, fastmath=True)
def EffBathParam(τc, ηc, ωc, N, num, λQ, γQ, ωQ, Λ):  
    α = 1/τc
    ωj = np.zeros((N))
    cj = np.zeros((N), dtype = np.complex128)

    # if num == False:
    # # ANALYTIC DISCRETIZATION OF THE DRUDE - LORENTZ SPECTRAL DENSITY 
    # # Huo. P., et al (Mol. Phys. 2012, 110, 1035–1052)
    #     arr = np.arange(0,N,1) + 1
    #     ω_max = 10 * γD
    #     ωj[:] = γD * np.tan(arr/N * np.arctan(ω_max/γD))
    #     cj[:] = 2 * ωj[:] * np.sqrt(λD * np.arctan(ω_max/γD)/(np.pi * N))
    
    # else:
    ω  = np.linspace(1E-10,100*ωc,50000)                   # FREQUENCY SCAN FOR BATH FREQUENCIES
    dω = ω[1] - ω[0]
    # NUMERICAL DISCRETIZATION OF SPECTRAL DENSITY
    # Walters, P. L.; et al.  J Comput Chem 2017, 38 (2), 110–115. https://doi.org/10.1002/jcc.24527.

    J = J_eff(α , ηc, ωc, ω,λQ,γQ,nsolvent,1,Λ,ωQ)  

    Fω = np.zeros(len(ω)) # LHS of eq. 2.6
    for i in range(len(ω)):
        Fω[i] = (4/np.pi) * np.sum(J[:i]/ω[:i]) * dω

    λs =  Fω[-1]
    for i in range(N):
        costfunc = np.abs(Fω-(((float(i)-0.5)/float(N))*λs)) # see eq. 2.6
        m = np.argmin((costfunc))
        ωj[i] = ω[m]
    mj = MQ # let all molecules have equal mass
    cj[:] = ωj[:] * np.sqrt(mj) * ((λs/(2*float(N)))**0.5) # see paragraph under eq 2.6 and earlier paragraph that says cj = κ*sqrt(m_j)*ω_j
    # print("CQ: ",cj[0]/cmtoau)
    return cj, ωj

# BOSONIC CREATION OPERATOR
@nb.jit(nopython=True, fastmath=True)
def creation(n):
    a = np.zeros((n,n), dtype = np.complex128)
    b = np.array([(x+1)**0.5 for x in range(n)], dtype = np.complex128)
    np.fill_diagonal(a[1:], b)
    return a

# ∂H/∂x_i - POSITION INDEPENDENT PART
# THIS FUNCTION CAN NOT BE JITTED!!!
def dHij_cons(cj):
    dHij         = np.zeros((ndof, nDW, nDW), dtype = np.complex128)
    dHij[:,:,:] -= np.kron(cj, R).T.reshape(ndof, nDW, nDW)
    return dHij

# ELECTRONIC HAMILTONIAN 
# CONTRUCTED IN THE IN THE DIABATIC BASIS WITH 4 VIBRATIONAL STATES |ν_L⟩, |ν_R⟩, |ν'_L⟩, |ν'_R⟩ 
@nb.jit(nopython=True, fastmath=True)
def Hel_cons(data):
    R2 = R @ R                                          # Rx^2
    H  = np.zeros((nDW, nDW), dtype = np.complex128)    
    H += np.diag(diaE)                                  # VIBRATIONAL STATES ENERGY | GROUND STATE ENERGY IS SUBSTRACTED
    H[0,1] += (EDW[1] - EDW[0])/2                       # |ν_L⟩ - |ν_R⟩ coupling    Δ  = (E[0] - E[1])/2
    H[1,0] += (EDW[1] - EDW[0])/2                       
    H[2,3] += (EDW[3] - EDW[2])/2                       # |ν'_L⟩ - |ν'_R⟩ coupling  Δ' = (E[2] - E[3])/2
    H[3,2] += (EDW[3] - EDW[2])/2                       
    H      += np.sum(data.cj[:]**2/data.ωj[:]**2) * R2/2      # Adds reorganization energy
    # if ~np.isfinite(H).all():
    #         raise Exception(f"Hel is not finite. Hel: {H} R2: {R2} rho: {data.ρt} nDW: {nDW}  diaE: {np.diag(diaE)} EDW: {EDW} cj: {data.cj} ωj: {data.ωj}")
    # H      += np.sum(data.cj[:data.nt]**2/data.ωj[:data.nt]**2) * R2/2      # Adds reorganization energy
    # H      += ηc**2*data.ωc   # add reorg energy of effective bath - if the results are wrong, it's probably because of this
    return H 

'''
    SIMULATION PARAMETERS 
        Hu. D., et al. (J. Phys. Chem. Lett. 2023, 14 (49), 11208–11216. https://doi.org/10.1021/acs.jpclett.3c02985.)
'''

# DEFINE cj and ωj
@nb.jit(nopython=True, fastmath=True)
def calc_cjωj(ωc):
    num    = True                                             # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical

    # # combine all lambda and gamma values so I don't have to calculate the bath over and over
    # λcomb = λD
    # γcomb = γD
    # for ii in range(nsolvent):
    #     λcomb = np.hstack((λcomb,λQ))

    cj, ωj = BathParam(λD, γD, ndofb, num)                      # BATH COUPLINGS AND FREQUENCIES
    # ck, ωk = BathParam(λQ, γQ, ndofs, num)                      # solvent BATH COUPLINGS AND FREQUENCIES

    # combine couplings and frequencies into one matrix
    # ck = np.repeat(ck,nsolvent) # get couplings for each solvent molecule
    # ωk = np.repeat(ωk,nsolvent)
    # cj = np.hstack((cj,ck))
    # ωj = np.hstack((ωj,ωk))
    # for ii in range(nsolvent):
    #     # combine bath parameters into one variable
    #     cj = np.hstack((cj,ck))
    #     ωj = np.hstack((ωj,ωk))

    if nbath > nsolvent+1: # include cavity if requested
        ck, ωk = EffBathParam(τc, ηc, ωc, ndofc, num, λQ, γQ, ωQ, Λ)                       # cavity BATH COUPLINGS AND FREQUENCIES

        print("check Λ: ", np.sum(ck[:]**2/(2*ωQ**2))/cmtoau, " ",Λ/cmtoau)

        # combine bath parameters into one variable
        cj = np.hstack((cj,ck))
        ωj = np.hstack((ωj,ωk))
        # print("check Λ including cj not just ck: ", np.sum(cj[:]**2/(2*ωQ**2))/cmtoau)
    return cj, ωj

# PHYSICAL CONSTANTS
# ==================================
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK  = 3.1577464e+05 
temp   = 300 / autoK
β      = 1 / temp 

# SYSTEM PARAMETERS ==================================
M = 1.0                                                   # R0 MASS
MQ = 1.0                                                  # Solvent MASS
nDW = 4                                                   # NUMBER OF VIBRATIONAL STATES IN DW - note that if I increase this, my diaE and diaV need to be changed too
wDW = 1000 * cmtoau                                       # DW BARRIER FREQUENCY
nSlevels = 3                                              # Number of levels represented in the solvent harmonic oscillator
N = 1024                                                  # NUMBER OF POINTS THAT DISCRETIZE R0 FOR DVR
L = 100.0                                                 # UPPER AND LOWER R0 LIMIT [-L, L]
x0 = np.linspace(-L,L,N)                                  # R0
dx = x0[0] - x0[1]                                        # dx
# xS0 = np.linspace(-L,L,N)                                 # Q0
# dSx = x0[0] - x0[1]                                       # dx for solvent
EDW, VDW = DVR(x0,M,wDW)                                  # EIGENENERGIES AND EIGENSTATES FOR THE DW (Nuclear kinetic energy plus the V potential)
Normx = np.trapz(VDW[:,0].conjugate() * VDW[:,0],x0,dx)    
VDW = VDW/(Normx)**0.5                                    # NORMALIZE THE EIGENSTATES
VDW = -1.0 * np.array(VDW, dtype = np.complex128)         # EIGENSTATES ARE IN THE OPPOSITE DIRECTION
# ES, VS = DVRS(x0,MQ,wDW)                                  # EIGENENERGIES AND EIGENSTATES FOR THE DW (Nuclear kinetic energy plus the V potential)
# Normx = np.trapz(VS[:,0].conjugate() * VS[:,0],x0,dx)    
# VS = VS/(Normx)**0.5                                      # NORMALIZE THE EIGENSTATES
# VS = np.array(VS, dtype = np.complex128)                  # make it complex

# DIABATIZATION OF  VIBRATIONAL STATES
# EIGENSTATES
diaV = np.zeros((len(VDW[:,0]), 4), dtype = np.complex128)
# |ν_L⟩ = (|0⟩ + |1⟩)/√2             |ν_R⟩ = (|0⟩ - |1⟩)/√2   
diaV[:,0], diaV[:,1] = (VDW[:,0] + VDW[:,1])/2**0.5, (VDW[:,0] - VDW[:,1])/2**0.5
# |ν'_L⟩ = (|2⟩ + |3⟩)/√2            |ν'_R⟩ = (|2⟩ - |3⟩)/√2   
diaV[:,2], diaV[:,3] = -(VDW[:,2] + VDW[:,3])/2**0.5, -(VDW[:,2] - VDW[:,3])/2**0.5
# EIGENENERGIES
diaE = np.zeros((4), dtype = np.complex128)
# E[ν_L] = E[ν_R] = (E[0] + E[1])/2 
diaE[0], diaE[1] = (EDW[0] + EDW[1])/2, (EDW[0] + EDW[1])/2
# E[ν'_L] = E[ν'_R] = (E[2] + E[3])/2 
diaE[2], diaE[3] = (EDW[2] + EDW[3])/2, (EDW[2] + EDW[3])/2
diaE -= diaE[0]                                                 # GROUND STATE ENERGY IS SUBSTRACTED
# POSITION OPERATOR
R = Rx(nDW)

# print("Transition Energy: ", diaE[2]/cmtoau)

# POSITION OPERATOR
# Q = Qx(nSlevels)

# INITIAL STATE ==================================
# SYSTEM IS INITIALIZED IN THE REACTANT STATE |ν_L⟩
ρ0 = np.zeros((nDW,nDW), dtype = np.complex128)
ρ0[0,0] = 1.0 + 0 * 1j

# SIMULATION PARAMETERS ==============================
parallel = True                                            # DO PARALLELIZATION
Cpus     = 100                                             # NUMBER THE CPUS USE FOR PARALLELIZATION
NTraj    = 3000                                           # NUMBER OF TRAJECTORIES
tf       = 10000 * fstoau                                   # SIMULATION TIME IN FEMTOSECONDS
dtN      = 6                                               # NUCLEAR TIME STEP
NSteps   = int(tf/dtN)                                     # NUMBER OF SIMULATION STEPS
Sim_time = np.array([(x * dtN) for x in range(NSteps)])    # SIMULATION TIMES ARRAY
Estep    = 30                                              # NUMBER OF ELECTRONIC STEPS PER NUCLEAR TIME STEP ⇒ MUST BE EVEN!!!!
dtE      = dtN/Estep                                       # ELECTRONIC TIME STEP
nskip    = 30                                               # FRAME SAVING RATE

if NSteps%nskip == 0:
    nData = NSteps // nskip + 0
else :
    nData = NSteps // nskip + 1

# BATH PARAMETERS ==============================
nsolvent = 1                                            # number of solvent molecules to simulate
nbath = nsolvent + 2    # number of baths present (solvent and cavity)
ndofs = 300                                                # number of frequencies per solvent molecule in discretization
ndofb   = 300                                              # NUMBER OF BATH OSCILLATORS (low frequencies of molecule?)
ndofc = 300                                                # number of cavity degrees of freedom
if nbath == nsolvent+1:
    ndofDWC = ndofb                                        # total number of degrees of freedom in the double well plus the cavity
    ndofSC = ndofs*nsolvent                                # degrees of freedom in one solvent molecule plus the cavity
else:
    ndofDWC = ndofb + ndofc
    ndofSC = ndofs*nsolvent + ndofc
ndof = ndofDWC#ndofs*nsolvent + ndofDWC
γD     = 200 * cmtoau                                      # BATH CHARACTERISTIC FREQUENCY (value from Sebastian's JACS paper)
η0 = 0.1
λD     = η0 * M * wDW * γD/2                                 # BATH REORGANIZATION ENERGY  (equation from Arkajit's paper?) 
# print("λD ",λD/cmtoau)
γQ     = 6000 * cmtoau                                     # Solvent bath CHARACTERISTIC FREQUENCY   (value from Sebastian's JACS paper) 
λQ     = 0.147 * cmtoau                                    # solvent BATH REORGANIZATION ENERGY  
ωQ     = 1189.7 * cmtoau                                   # solvent characteristic frequency
Λ      = 0.0009328299310150123*cmtoau#1.71 *cmtoau                                      # spectator mode reorganization energy
num    = False                                             # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical

τc = 500*fstoau
Ω = 114 # Rabi Splitting
ηc = 0.005*Ω/114.05702851425713 #au - change to 0 for no cavity

# TIME INDEPENDENT FUNCTIONS ==============================
# Hel  = Hel_cons(cj,ωj)                                          # ELECTRONIC HAMILTONIAN | INDEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
# dHij = dHij_cons(cj)                                         # ∂H/∂x_i                | INDEPENDENT OF THE POSITION OF THE BATH OSCILLATOR | DO NOT JIT


# if __name__ == '__main__': 
# print(np.real(np.round(Hel/cmtoau,3)))
# print('================')
# print(np.real(np.round(R,3)))
# print('================')
# print(np.real(np.sum((cj/ωj)**2)*np.dot(R,R)/2)/cmtoau)
# J = J_eff(1/τc, ηc, 1190, np.linspace(0,2500,200)*cmtoau,λQ,γQ,nsolvent,1,Λ,ωQ)
# for n in range(ndof):
#     plt.axvline(ωj[n]/cmtoau, ls = '-.', color = 'black', lw = 1)
# plt.plot(np.linspace(0,2500,200),J, lw = 3, c = 'r')
# plt.savefig('images/spectralDen.png')

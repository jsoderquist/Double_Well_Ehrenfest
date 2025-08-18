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

# attempt to create a kronecker product that will jit
# @nb.jit(nopython=True, fastmath=True)
# def kron2D(a,b):
#     dim = a.shape[0] # assumes matrices are square and the same shape and doubles
#     c = np.zeros((dim**2,dim**2), dtype = np.float64)

#     for ii in range(dim):
#         for jj in range(dim):
#             for kk in range(dim):
#                 c[dim*ii+kk,dim*jj:dim*(jj+1)] = a[ii,jj]*b[kk,:]
#     return c
    

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
@nb.jit(nopython=True, fastmath=True)
def Qx(nlevels,V,x0,dx):
    pos = np.zeros((nlevels,nlevels), dtype = np.complex128)
    for j in range(nlevels):
        for i in range(nlevels):
            avg_pos = V[:,j].conjugate() * x0 * V[:,i]
            pos[j,i] = np.trapz(avg_pos,x0,dx)
    return pos

# DOUBLE WELL POTENTIAL ENERGY
@nb.jit(nopython=True, fastmath=True)
def DW(x,m,wDW):
    Eb = 2250 * cmtoau
    V = -(m*wDW**2 / 2) * x**2 + (m**2*wDW**4 / (16 * Eb)) * x**4
    return V - min(V)

# harmonic potential
@nb.jit(nopython=True, fastmath=True)
def harmonicPot(x,m,ω):
    m = 1 # I don't know why this is being redefined here. It's already given to the function as 1, but it's not used anyway
    V = 0.5*m*ω**2*x**2
    return V - min(V)

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
@nb.jit(nopython=True, fastmath=True)
def DVRS(x,m,ω):
    V = harmonicPot(x,m,ω)
    V = np.diag(V)
    K = T(x,m)
    E, V = np.linalg.eigh(V+K)
    return E, V

# DRUDE - LORENTZ SPECTRAL DENSITY
@nb.jit(nopython=True, fastmath=True)
def J_DrudeL(λ, γ, ω):
    return (2 * γ * λ * ω) / (ω**2 + γ**2)

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

# # BOSONIC CREATION OPERATOR
# @nb.jit(nopython=True, fastmath=True)
# def creation(n):
#     a = np.zeros((n,n), dtype = np.complex128)
#     b = np.array([(x+1)**0.5 for x in range(n)], dtype = np.complex128)
#     np.fill_diagonal(a[1:], b)
#     return a

# ∂H/∂x_i - POSITION INDEPENDENT PART
# THIS FUNCTION CAN NOT BE JITTED!!!
def dHij_cons(cj):
    dHij         = np.zeros((ndof, hamDim, hamDim), dtype = np.complex128)
    Rcoupling = np.kron(cj[:ndofb], Rextended).T.reshape(ndofb, hamDim, hamDim) # parts of force with R
    Qcoupling = np.kron(cj[ndofb:], Qextended).T.reshape(ndof-ndofb,hamDim, hamDim) # parts of force with Q
    dHij -= np.concatenate((Rcoupling, Qcoupling), axis=0)
    return dHij

# ELECTRONIC HAMILTONIAN - includes V and T of both the molecule and solvents as well as the reorganization energies
# CONTRUCTED IN THE IN THE DIABATIC BASIS WITH 4 VIBRATIONAL STATES |ν_L⟩, |ν_R⟩, |ν'_L⟩, |ν'_R⟩ 
# @nb.jit(nopython=True, fastmath=True)
def Hel_cons(data,ωc):
    R2 = R @ R                                          # Rx^2
    Q2 = Q @ Q                                          # Q^2
    IR = np.eye(data.nt) # identity matrix of R
    IQ = np.eye(data.nSlevels) # identity matrix of Q

    # molecule V+T part of Hamiltonian
    HM  = np.zeros((nDW, nDW), dtype = np.complex128)    
    HM += np.diag(diaE)                                  # VIBRATIONAL STATES ENERGY | GROUND STATE ENERGY IS SUBSTRACTED
    HM[0,1] += (EDW[1] - EDW[0])/2                       # |ν_L⟩ - |ν_R⟩ coupling    Δ  = (E[0] - E[1])/2
    HM[1,0] += (EDW[1] - EDW[0])/2                       
    HM[2,3] += (EDW[3] - EDW[2])/2                       # |ν'_L⟩ - |ν'_R⟩ coupling  Δ' = (E[2] - E[3])/2
    HM[3,2] += (EDW[3] - EDW[2])/2

    # Solvent V+T part of Hamiltonian
    HS  = np.zeros((nSlevels, nSlevels), dtype = np.complex128)   
    HS += np.diag(ES)                                  # VIBRATIONAL STATES ENERGY | GROUND STATE ENERGY IS SUBSTRACTED

    # Cavity V+T part of Hamiltonian
    Hc  = np.zeros((nPhLevels, nPhLevels), dtype = np.complex128)   
    Hc += np.diag(data.Ec)                                  # VIBRATIONAL STATES ENERGY | GROUND STATE ENERGY IS SUBSTRACTED

    # Add R-dependent part of H_Q (expand H_Q to see this)
    HM += np.sum(cQ**2/ωQ**2) * R2/2

    # add reorganization energies                 
    HM      += np.sum(data.cj[:ndofb]**2/data.ωj[:ndofb]**2) * R2/2      # Adds reorganization energy of molecule bath
    HS      += (np.sum(data.cj[ndofb:]**2/data.ωj[ndofb:]**2)/2 \
        + ηc**2*data.ωc) * Q2      # Adds reorganization energy of solvent bath and cavity (equation S12)

    # Connect R and Q portions into one hamiltonian - done last to do less tensor products
    H = np.kron(np.kron(HM,Ic),IQ) + np.kron(np.kron(IR,Ic),HS)
    # this next part is currently only one molecule - adds final term of H_Q
    H -= np.sum(data.cj[ndofb:(ndofb+ndofs)])*Qextended@Rextended

    # include solvent-cavity coupling term
    H += ωc**2*np.sqrt(2/ωc)*ηc*data.qcextended@Qextended

    return H 

'''
    SIMULATION PARAMETERS 
        Hu. D., et al. (J. Phys. Chem. Lett. 2023, 14 (49), 11208–11216. https://doi.org/10.1021/acs.jpclett.3c02985.)
'''

# DEFINE cj and ωj
@nb.jit(nopython=True, fastmath=True)
def calc_cjωj(ωc):
    num    = True                                             # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical

    cj, ωj = BathParam(λD, γD, ndofb, num)                      # BATH COUPLINGS AND FREQUENCIES
    cs, ωs = BathParam(λQ, γQ, ndofs, num)                      # Solvent BATH COUPLINGS AND FREQUENCIES
    # ck, ωk = EffBathParam(τc, ηc, ωc, ndofc, λQ, γQ, ωQ, Λ)                       # cavity BATH COUPLINGS AND FREQUENCIES

    # combine bath parameters into one variable
    # order: molecule bath, cavity bath, solvent
    # cj = np.hstack((cj,ck))
    # ωj = np.hstack((ωj,ωk))
    cj = np.hstack((cj,cs))
    ωj = np.hstack((ωj,ωs))

    return cj, ωj

# prepare cavity parameters that are dependent on ωc and thus have to be computed for each run
def prepare_cavity(data,ωc):
    Ec, Vc = DVRS(xc0,1,ωc)                                  # EIGENENERGIES AND EIGENSTATES FOR THE DW (Nuclear kinetic energy plus the V potential)
    Normx = np.trapz(Vc[:,0].conjugate() * Vc[:,0],xc0,dcx)    
    Vc = Vc/(Normx)**0.5                                      # NORMALIZE THE EIGENSTATES
    Vc = np.array(Vc, dtype = np.complex128)                  # make it complex
    data.qc = Qx(nPhLevels,Vc,xc0,dcx) # cavity
    data.Ec = Ec[:nPhLevels]
    


# PHYSICAL CONSTANTS
# ==================================
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK  = 3.1577464e+05 
temp   = 300 / autoK
β      = 1 / temp 

# BATH PARAMETERS ==============================
M = 1.0                                                   # R0 MASS
MS = 1.0                                                  # Solvent MASS
nDW = 4                                                   # NUMBER OF VIBRATIONAL STATES IN DW - note that if I increase this, my diaE and diaV need to be changed too
wDW = 1000 * cmtoau                                       # DW BARRIER FREQUENCY
nSlevels = 2                                              # Number of levels represented in the solvent harmonic oscillator
nsolvent = 1                                            # number of solvent molecules to simulate
nPhLevels = 2                                             # number of photonic levels
hamDim = nDW*nSlevels**nsolvent*nPhLevels                  # dimensionality of the Hamiltonian
ndofs = 300                                                # number of frequencies per solvent molecule in discretization
ndofb   = 300                                              # NUMBER OF BATH OSCILLATORS (low frequencies of molecule?)
# ndofc = 300                                                # number of cavity degrees of freedom (really how many discrete points we take in Jeff)
ndof = ndofb + nsolvent*ndofs
γD     = 200 * cmtoau                                      # BATH CHARACTERISTIC FREQUENCY (value from Sebastian's JACS paper)
η0 = 0.1
λD     = η0 * M * wDW * γD/2                                 # BATH REORGANIZATION ENERGY  (equation from Arkajit's paper?) 
γQ     = 6000 * cmtoau                                     # Solvent bath CHARACTERISTIC FREQUENCY   (value from Sebastian's JACS paper) 
λQ     = 0.147 * cmtoau                                    # solvent BATH REORGANIZATION ENERGY  
ωQ     = 1189.7 * cmtoau                                   # solvent characteristic frequency
cQ     = 0.110 * cmtoau                                    # solvent-reactant coupling (should be different for every solvent molecule, but it's all the same in Sebastian's model)
Λ      = 0.0009328299310150123*cmtoau#1.71 *cmtoau                                      # spectator mode reorganization energy
num    = False                                             # DISCRETIZATION OF THE SPECTRAL DENSITY | True ⇒ Numerical | False ⇒ Analytical

τc = 500*fstoau
Ω = 114 # Rabi Splitting
ηc = 0.005*Ω/114.05702851425713 #au - change to 0 for no cavity

# SYSTEM PARAMETERS ==================================
N = 1024                                                  # NUMBER OF POINTS THAT DISCRETIZE R0 FOR DVR
L = 100.0                                                 # UPPER AND LOWER R0 LIMIT [-L, L]
x0 = np.linspace(-L,L,N)                                  # R0
dx = abs(x0[0] - x0[1])                                   # dx
xS0 = np.linspace(-L,L,N)                                 # Q0
dSx = abs(xS0[0] - xS0[1])                                # dx for solvent
xc0 = np.linspace(-L,L,N)                                 # Q0
dcx = abs(xc0[0] - xc0[1])                                # dx for solvent
EDW, VDW = DVR(x0,M,wDW)                                  # EIGENENERGIES AND EIGENSTATES FOR THE DW (Nuclear kinetic energy plus the V potential)
Normx = np.trapz(VDW[:,0].conjugate() * VDW[:,0],x0,dx)    
VDW = VDW/(Normx)**0.5                                    # NORMALIZE THE EIGENSTATES
VDW = -1.0 * np.array(VDW, dtype = np.complex128)         # EIGENSTATES ARE IN THE OPPOSITE DIRECTION

# DIABATIZATION OF  VIBRATIONAL STATES
# EIGENSTATES
diaV = np.zeros((len(VDW[:,0]), nDW), dtype = np.complex128)
# |ν_L⟩ = (|0⟩ + |1⟩)/√2             |ν_R⟩ = (|0⟩ - |1⟩)/√2   
diaV[:,0], diaV[:,1] = (VDW[:,0] + VDW[:,1])/2**0.5, (VDW[:,0] - VDW[:,1])/2**0.5
# |ν'_L⟩ = (|2⟩ + |3⟩)/√2            |ν'_R⟩ = (|2⟩ - |3⟩)/√2   
diaV[:,2], diaV[:,3] = -(VDW[:,2] + VDW[:,3])/2**0.5, -(VDW[:,2] - VDW[:,3])/2**0.5
# EIGENENERGIES
diaE = np.zeros((nDW), dtype = np.complex128)
# E[ν_L] = E[ν_R] = (E[0] + E[1])/2 
diaE[0], diaE[1] = (EDW[0] + EDW[1])/2, (EDW[0] + EDW[1])/2
# E[ν'_L] = E[ν'_R] = (E[2] + E[3])/2 
diaE[2], diaE[3] = (EDW[2] + EDW[3])/2, (EDW[2] + EDW[3])/2
if nDW > 4:                         # get states above the well if requested
    diaV[:,5:] = VDW[:,5:nDW]
    diaE[5:] = EDW[5:nDW]
diaE -= diaE[0]                                                 # GROUND STATE ENERGY IS SUBSTRACTED
ES, VS = DVRS(xS0,MS,ωQ)                                  # EIGENENERGIES AND EIGENSTATES FOR THE DW (Nuclear kinetic energy plus the V potential)
Normx = np.trapz(VS[:,0].conjugate() * VS[:,0],xS0,dSx)    
VS = VS/(Normx)**0.5                                      # NORMALIZE THE EIGENSTATES
VS = np.array(VS, dtype = np.complex128)                  # make it complex


# POSITION OPERATORS
R = Rx(nDW) # MOLECULE
Q = Qx(nSlevels,VS,xS0,dSx) # SOLVENT

# Only include the states we requested - note that this has to be done after calculating Q
ES = ES[:nSlevels]
VS = VS[:nSlevels]

# INITIAL STATE ==================================
# SYSTEM IS INITIALIZED IN THE REACTANT STATE |ν_L⟩
IR = np.eye(nDW) # identity matrix of R
IQ = np.eye(nSlevels) # identity matrix of Q
Ic = np.eye(nPhLevels) # identity matrix of qc
Rextended = np.kron(np.kron(R,Ic),IQ)
Qextended = np.kron(np.kron(IR,Ic),Q)
ρR0 = np.zeros((nDW,nDW), dtype = np.complex128)
ρR0[0,0] = 1.0 + 0 * 1j
ρS0 = np.zeros((nSlevels,nSlevels), dtype = np.complex128)
ρS0[0,0] = 1.0 + 0 * 1j
ρc0 = np.zeros((nPhLevels,nPhLevels), dtype = np.complex128)
ρc0[0,0] = 1.0 + 0 * 1j
ρ0 = np.kron(np.kron(ρR0,ρc0),ρS0)

# SIMULATION PARAMETERS ==============================
parallel = True                                            # DO PARALLELIZATION
Cpus     = 100                                             # NUMBER THE CPUS USE FOR PARALLELIZATION
NTraj    = 2                                           # NUMBER OF TRAJECTORIES
tf       = 20 * fstoau                                   # SIMULATION TIME IN FEMTOSECONDS
dtN      = 1                                               # NUCLEAR TIME STEP
NSteps   = int(tf/dtN)                                     # NUMBER OF SIMULATION STEPS
Sim_time = np.array([(x * dtN) for x in range(NSteps)])    # SIMULATION TIMES ARRAY
Estep    = 30                                              # NUMBER OF ELECTRONIC STEPS PER NUCLEAR TIME STEP ⇒ MUST BE EVEN!!!!
dtE      = dtN/Estep                                       # ELECTRONIC TIME STEP
nskip    = 30                                               # FRAME SAVING RATE

if NSteps%nskip == 0:
    nData = NSteps // nskip + 0
else :
    nData = NSteps // nskip + 1


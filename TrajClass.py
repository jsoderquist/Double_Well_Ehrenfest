import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit
import parameters as par
# ==================================

spec = [
    ('nt',                      int32), # NUMBER OF VIBRATIONAL STATES 
    ('nSteps',                  int32), # NUMBER OF EVOLUTION STEPS
    ('nData',                   int32), # NUMBER OF SAVED STEPS
    ('ndof',                    int32), # NUMBER OF BATH MODES
    ('ndofb',                   int32), # NUMBER OF molecule BATH MODES
    ('ndofs',                   int32), # NUMBER OF solvent BATH MODES
    ('nsolvent',                int32), # NUMBER OF solvent molecules to simulate
    ('nSlevels',                int32), # NUMBER OF energy levels in solvent
    ('nPhLevels',               int32), # NUMBER OF energy levels in cavity
    ('hamDim',                  int32), # dimensionality of the Hamiltonian
    ('ωc',                    float64), # current cavity resonant frequency
    ('x',                  float64[:]), # BATH POSITION
    ('P',                  float64[:]), # BATH MOMENTA
    ('v',                  float64[:]), # BATH VELOCITY
    ('F1',                 float64[:]), # BATH FORCE AT t
    ('F2',                 float64[:]), # BATH FORCE AT t + 1
    ('ρt',            complex128[:,:]), # DENSITY MATRIX AT TIME t
    ('H_bc',          complex128[:,:]), # Bath coupling HAMILTONIAN | DEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
    ('H_el',          complex128[:,:]), # ELECTRONIC HAMILTONIAN
    ('ρw',            float64[:,:]), # PLACE HOLDER FOR THE DENSITY MATRIX
    # ('Rextended',             float64[:,:]), # Position coordinate of the molecule in extended basis
    # ('Qextended',             float64[:,:]), # Position coordinate of the solvent in extended basis
    ('qc',             float64[:,:]), # Position coordinate of the cavity
    ('Ec',                     float64[:]), # cavity energies
    ('test',          complex128[:,:]), # PLACE HOLDER FOR THE DENSITY MATRIX
    ('cj',            complex128[:]), # place holder for the coupling coefficients
    ('ωj',                 float64[:]), # place holder for the discretized frequencies
    ('dHij',          complex128[:,:,:]), # place holder
]

@jitclass(spec)
class trajData(object):
    def __init__(self, nDW, ndof, nSteps, nData, ωc, ndofb, ndofs, nsolvent, nSlevels,nPhLevels,hamDim):
        self.nt     = nDW
        self.nSteps = nSteps
        self.nData  = nData
        self.ndof   = ndof
        self.ndofb   = ndofb
        self.ndofs   = ndofs
        self.nsolvent   = nsolvent
        self.nSlevels = nSlevels
        self.nPhLevels = nPhLevels
        self.hamDim = hamDim
        self.ωc     = ωc
        self.x      = np.zeros(self.ndof, dtype = np.float64)
        self.P      = np.zeros(self.ndof, dtype = np.float64)
        self.v      = np.zeros(self.ndof, dtype = np.float64)
        self.F1     = np.zeros(self.ndof, dtype = np.float64)
        self.F2     = np.zeros(self.ndof, dtype = np.float64)
        self.ρt     = np.zeros((hamDim,hamDim), dtype = np.complex128)
        self.H_bc   = np.zeros((hamDim,hamDim), dtype = np.complex128)
        self.H_el   = np.zeros((hamDim,hamDim), dtype = np.complex128)
        self.ρw     = np.zeros((nData,self.nt))
        # self.Rextended = np.zeros((self.nt*self.nSlevels,self.nt*self.nSlevels), dtype = np.float64)
        # self.Qextended = np.zeros((self.nt*self.nSlevels,self.nt*self.nSlevels), dtype = np.float64)
        self.qc   = np.zeros((par.nPhLevels,par.nPhLevels), dtype = np.float64)
        self.Ec     = np.zeros(par.nPhLevels, dtype = np.float64)
        self.test   = np.zeros((nData,2) , dtype = np.complex128)
        self.cj     = np.zeros(self.ndof, dtype = np.complex128)
        self.ωj     = np.zeros(self.ndof, dtype = np.float64)
        self.dHij   = np.zeros((self.ndof, hamDim, hamDim), dtype = np.complex128)
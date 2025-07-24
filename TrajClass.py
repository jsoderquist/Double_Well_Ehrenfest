import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit
import parameters as par
# ==================================

spec = [
    ('nt',                      int32), # NUMBER OF VIBRATIONAL STATES 
    ('ns'                       int32), # number of solvent energy levels
    ('nSteps',                  int32), # NUMBER OF EVOLUTION STEPS
    ('nData',                   int32), # NUMBER OF SAVED STEPS
    ('ndofDWC',                 int32), # NUMBER OF BATH MODES (double well + cavity)
    ('ndofSC',                  int32), # NUMBER OF BATH MODES (solvent + cavity)
    ('nsolvent',                int32), # Number of solvent molecules to simulate
    ('x',                  float64[:]), # BATH POSITION
    ('P',                  float64[:]), # BATH MOMENTA
    ('v',                  float64[:]), # BATH VELOCITY
    ('F1',                 float64[:]), # BATH FORCE AT t
    ('F2',                 float64[:]), # BATH FORCE AT t + 1
    ('ρt',            complex128[:,:]), # DENSITY MATRIX AT TIME t
    ('H_bc',          complex128[:,:]), # ELECTRONIC HAMILTONIAN | DEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
    ('ρw',            float64[:,:]),    # PLACE HOLDER FOR THE DENSITY MATRIX
    ('test',          complex128[:,:]), # PLACE HOLDER FOR THE DENSITY MATRIX
    ('cjDWC',           complex128[:]), # place holder for the coupling coefficients of the double well + cavity
    ('ωjDWC',              float64[:]), # place holder for the discretized frequencies of the double well + cavity
    ('cjSC',           complex128[:,:]), # place holder for the coupling coefficients of the solvents + cavity
    ('ωjSC',              float64[:,:]), # place holder for the discretized frequencies of the solvents + cavity
]

@jitclass(spec)
class trajData(object):
    def __init__(self, nDW, nSlevels, ndofDWC, ndofSC, nSteps, nData, nsolvent):
        self.nt     = nDW
        self.ns     = nSlevels
        self.nSteps = nSteps
        self.nData  = nData
        self.ndofDWC   = ndofDWC
        self.ndofSC   = ndofSC
        self.nsolvent = nsolvent
        dims = self.nt*self.ns**self.nsolvent # dimensionality of the Hilbert space
        self.x      = np.zeros(self.ndofDWC, dtype = np.float64)
        self.P      = np.zeros(self.ndofDWC, dtype = np.float64)
        self.v      = np.zeros(self.ndofDWC, dtype = np.float64)
        self.F1     = np.zeros(self.ndofDWC, dtype = np.float64)
        self.F2     = np.zeros(self.ndofDWC, dtype = np.float64)
        self.ρt     = np.zeros((dims,dims), dtype = np.complex128)
        self.H_bc   = np.zeros((dims,dims), dtype = np.complex128)
        self.ρw     = np.zeros((nData,self.nt)) # this only holds populations of the double well at each time?
        self.test   = np.zeros((nData,2) , dtype = np.complex128)
        self.cjDWC     = np.zeros(self.ndofDWC, dtype = np.complex128)
        self.ωjDWC     = np.zeros(self.ndofDWC, dtype = np.float64)
        self.cjSC     = np.zeros((self.ndofSC,self.nsolvent), dtype = np.complex128)
        self.ωjSC     = np.zeros((self.ndofSC,self.nsolvent), dtype = np.float64)

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
    ('ndofc',                   int32), # NUMBER OF cavity BATH MODES
    ('ndofs',                   int32), # NUMBER OF solvent BATH MODES
    ('nsolvent',                int32), # NUMBER OF solvent molecules to simulate
    ('ωc',                    float64), # current cavity resonant frequency
    ('x',                  float64[:]), # BATH POSITION
    ('P',                  float64[:]), # BATH MOMENTA
    ('v',                  float64[:]), # BATH VELOCITY
    ('F1',                 float64[:]), # BATH FORCE AT t
    ('F2',                 float64[:]), # BATH FORCE AT t + 1
    ('ρt',            complex128[:,:]), # DENSITY MATRIX AT TIME t
    ('H_bc',          complex128[:,:]), # ELECTRONIC HAMILTONIAN | DEPENDENT OF THE POSITION OF THE BATH OSCILLATOR
    ('ρw',            float64[:,:]), # PLACE HOLDER FOR THE DENSITY MATRIX
    ('test',          complex128[:,:]), # PLACE HOLDER FOR THE DENSITY MATRIX
    ('cj',            complex128[:]), # place holder for the coupling coefficients
    ('ωj',                 float64[:]), # place holder for the discretized frequencies
    ('dHij',          complex128[:,:,:]), # place holder
]

@jitclass(spec)
class trajData(object):
    def __init__(self, nDW, ndof, nSteps, nData, ωc, ndofb, ndofc, ndofs, nsolvent):
        self.nt     = nDW
        self.nSteps = nSteps
        self.nData  = nData
        self.ndof   = ndof
        self.ndofb   = ndofb
        self.ndofc   = ndofc
        self.ndofs   = ndofs
        self.nsolvent   = nsolvent
        self.ωc     = ωc
        self.x      = np.zeros(self.ndof, dtype = np.float64)
        self.P      = np.zeros(self.ndof, dtype = np.float64)
        self.v      = np.zeros(self.ndof, dtype = np.float64)
        self.F1     = np.zeros(self.ndof, dtype = np.float64)
        self.F2     = np.zeros(self.ndof, dtype = np.float64)
        self.ρt     = np.zeros((self.nt, self.nt), dtype = np.complex128)
        self.H_bc   = np.zeros((self.nt,self.nt), dtype = np.complex128)
        self.ρw     = np.zeros((nData,self.nt))
        self.test   = np.zeros((nData,2) , dtype = np.complex128)
        self.cj     = np.zeros(self.ndof, dtype = np.complex128)
        self.ωj     = np.zeros(self.ndof, dtype = np.float64)
        self.dHij   = np.zeros((ndof, nDW, nDW), dtype = np.complex128)
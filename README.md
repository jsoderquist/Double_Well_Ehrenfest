# Double Well Dynamics with Mean Field Ehrenfest

MFE propagation of the density matrix for a system in a Double-Well (DW) potential coupled to a bosonic bath.

$\hat{H} = \hat{T} + \hat{V}(\hat{R}_0) + \sum_j^N \left[\frac{\hat{p}_j^2}{2} + \frac{\omega^2_j}{2}\left(x_j - \frac{c_j}{\omega_j^2}\hat{R}_0 \right)^2 \right]$

With potential

$\hat{V}(\hat{R}_0) = \frac{M^2\omega_b^4}{16E_b}\hat{R}_0^4 - \frac{M\omega_b^2}{2}\hat{R}^2_0$ 

Here, $\hat{R}_0$ is the DW coordinate, $M$ is the effective mass, $\omega_b$ is the frequency of the top of the DW barrier and $E_b$ is the barrier height of the DW.

The vibrational eigenstates of the potential are obtained by the Discrete Variable Representation (DVR) basis. We consider only the four lowest-energy states which are diabatized as:

$|\nu_L⟩  = \frac{|\nu_0⟩ + |\nu_1⟩}{\sqrt{2}}, \qquad |\nu_R⟩  = \frac{|\nu_0⟩ - |\nu_1⟩}{\sqrt{2}}$

$|\nu'_L⟩ = \frac{|\nu_2⟩ + |\nu_3⟩}{\sqrt{2}}, \qquad |\nu'_R⟩ = \frac{|\nu_2⟩ - |\nu_3⟩}{\sqrt{2}}$

Where $|\nu_L⟩, |\nu_R⟩$ are the diabatic ground state vibrational states of the left and right side of the DW, while $|\nu'_L⟩, |\nu'_R⟩$ are the first excited vibrational states. Under this representation, the position coordinate is given by $\hat{R}_{ij} = ⟨\nu_i|\hat{R}_0|\nu_j⟩$.

For the Ehrenfest dynamics, the electronic subsystem is propagated via the von Neuman Equation

$\frac{\partial \rho(t)}{\partial t} = -\frac{i}{\hbar} \left[\hat{H}_{el},\rho(t)\right]$

Where the electronic Hamiltonian is given as

$\hat{H}_{el} = ⟨\nu_i|\hat{V}|\nu_j⟩ - \left(\sum_k c_k\hat{x}\right) \cdot \hat{R}_{ij} + \sum_k\frac{c^2_k}{2\omega_k^2} \hat{R}_{ij}^2$

With $\hat{R}_{ij}^2 = ⟨\nu_i|\hat{R}^2_0|\nu_j⟩$. The von Neuman equation is solved via the Runge-Kutta 4 algorithm.

The nuclear DOF are evolved according to the Newton's equation of motion via the Velocity Verled algorithm, with force computed as

$F_k = - \frac{\partial H}{\partial x_k} = -\omega_k^2 + \sum_{i,j} \rho_{ij} \cdot c_k $.

================================================================

The code uses the numba library to speed up the computations. The function dHij_cons() of the "parameters" file CAN NOT be jitted, as it produces incorrect results.

Here, there is a brief description of the simulation files.

### parameters.py

Contains all the time-independent parameters used in the simulation; it creates the bath position independent part of $\hat{H}_{el}$ and $  \frac{\partial H}{\partial x_k} $.

### model.py

Constains the bath position dependent part of $\hat{H}_{el}$ and the initialization of the bath DOFs.

### TrajClass

Contains all the variables that will be updated through out the simulation.

### MFE

Contains the functions involved in the time evolution of the system.

### dynamics

Initialize the simulation and allows the parallelization.

The code is parallelized through the multipar.py file. The results presented in the "images" folder reproduce the ones presented in Fig. S6a and Fig S6b of:

 Hu. D., et al. (J. Phys. Chem. Lett. 2023, 14 (49), 11208–11216. https://doi.org/10.1021/acs.jpclett.3c02985.)

 ### How to Change Parameters
 dynamics.py: change $\omega_c$ values to change data points to use (note: also update plotting_rho.py with the same values of $\omega_c$ but also include an $\omega_c$ of 0)
 
 parameters.py: multiply value in $eta_c$ by 0 to simulate outside the cavity. Ntraj, cpus, tf, and dtN may also be changed.
 
 multipar.py: change NARRAY to str(N-1) where N is the number of jobs desired

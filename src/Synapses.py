# -*- coding: utf-8 -*-

import numpy as np

class SingleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=5e-3):
        """
        Args:
            td (float):Synaptic decay time
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.r = np.zeros(N)

    def initialize_states(self):
        self.r = np.zeros(self.N)

    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td) + spike/self.td
        self.r = r
        return r
     

class DoubleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
    
    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*(1-self.dt/self.td) + spike/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r    #[pA]

class tsodyks_markram:
    def __init__(self, N, dt=0.0001, tau_rec=0.8, tau_inact=0.003, U=0.6):
        """
        Args:
            tau_rec (float): Synaptic recovery time constant
            tau_facil (float): Synaptic facilitation time constant
            U (float): Facilitation factor
        """
        self.N = N
        self.dt = dt
        self.tau_rec = np.full(N, tau_rec)
        self.tau_inact = np.full(N, tau_inact)
        self.U = np.full(N, U)
        self.x = np.full(N, 1.0)
        self.z = np.zeros(N)
        self.y = np.zeros(N)

    def __call__(self, spike):
        """
        spike_train: 1D array of 0 or 1 (1=spike)
        dt: time step (ms)
        Returns: synaptic output over time
        """
    
        dx = self.z/self.tau_rec * self.dt
        dy = -self.y/self.tau_inact * self.dt
        dz = (self.y/self.tau_inact - self.z/self.tau_rec) * self.dt

        idx = np.where(spike == 1)[0]
        if len(idx) > 0:
            dx[idx] -= self.U[idx] * self.x[idx]
            dy[idx] += self.U[idx] * self.x[idx]
            # print("------------------------")
            # print("Synaptic output:", self.y[0]+dy[0])
        
        self.x += dx
        self.y += dy
        self.z += dz
        
        return self.y
    
# class tsodyks_markram:
#     def __init__(self, N, dt=0.0001, tau_rec=1e-2, tau_facil=5e-3, U=0.5):
#         """
#         Args:
#             tau_rec (float): Synaptic recovery time constant
#             tau_facil (float): Synaptic facilitation time constant
#             U (float): Facilitation factor
#         """
#         self.N = N
#         self.dt = dt
#         self.tau_rec = tau_rec
#         self.tau_facil = tau_facil
#         self.U = U
#         self.x = np.full(N, 1.0)
#         self.u = np.full(N, U)
#         self.y = np.zeros(N)
#         self.decay_u = np.exp(-dt / tau_facil)
#         self.decay_x = np.exp(-dt / tau_rec)

#     def __call__(self, spike):
#         """
#         spike_train: 1D array of 0 or 1 (1=spike)
#         dt: time step (ms)
#         Returns: synaptic output over time
#         """
    
#         self.u = self.u * self.decay_u
#         self.x = self.x + (1 - self.x) * (1 - self.decay_x)

#         self.y[:] = 0.0
#         idx = np.where(spike == 1)[0]
#         if len(idx) > 0:
#             self.u[idx] += self.U * (1.0 - self.u[idx])
#             self.y[idx] = self.u[idx] * self.x[idx]
#             self.x[idx] -= self.y[idx]
#             if 0 in idx:
#                 print("Synaptic output:", self.y[0])
#         return self.y
    
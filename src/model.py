import numpy as np
class PQNModel:
    # available mode list
    MODE_LIST=['RSexci', 'RSinhi', 'FS', 'LTS', 'IB', 'EB', 'PB', 'Class2']

    # init
    def __init__(self, mode='RSexci', N = 1):
        self.N = N
        self.PARAM={}
        self.Y={}
        self.state_variable_v=np.zeros(N)
        self.state_variable_n=np.zeros(N)
        self.state_variable_q=np.zeros(N)
        self.state_variable_u=np.zeros(N)
        self.set_mode(mode)

    # set mode
    def set_mode(self, mode):
        if (mode in self.MODE_LIST) is False:
            raise ValueError('mode should be RSexci, RSinhi, FS, LTS, IB, EB, PB, or Class2')
        self.mode=mode
        self.set_PARAM()
        self.set_Y()

    # set parameters of the given mode
    def set_PARAM(self):
        self.PARAM['dt']=np.full(self.N, 0.0001)
        self.PARAM['afn']=np.full(self.N, 1.5625)
        self.PARAM['afp']=np.full(self.N, -0.5625)
        self.PARAM['bfn']=np.full(self.N, -1.125)
        self.PARAM['cfn']=np.full(self.N, 0)
        self.PARAM['agn']=np.full(self.N, 1)
        self.PARAM['agp']=np.full(self.N, 10.28125)
        self.PARAM['bgn']=np.full(self.N, 0.40625)
        self.PARAM['cgn']=np.full(self.N, 0)
        self.PARAM['ahn']=np.full(self.N, 0.28125)
        self.PARAM['ahp']=np.full(self.N, 9.125)
        self.PARAM['bhn']=np.full(self.N, -7.1875)
        self.PARAM['chn']=np.full(self.N, -2.8125)
        self.PARAM['tau']=np.full(self.N, 0.0064)
        self.PARAM['I0']=np.full(self.N, 2.375)
        self.PARAM['k']=np.full(self.N, 36.4375)
        self.PARAM['phi']=np.full(self.N, 4.75)
        self.PARAM['epsq']=np.full(self.N, 0.0693359375)
        self.PARAM['rg']=np.full(self.N, 0.0625)
        self.PARAM['rh']=np.full(self.N, 15.71875)
        self.state_variable_v=np.full(self.N, -4906)
        self.state_variable_n=np.full(self.N, 27584)
        self.state_variable_q=np.full(self.N, -3692)
        self.BIT_WIDTH=18
        self.BIT_WIDTH_FRACTIONAL=10
        self.BIT_Y_SHIFT=20
        
        self.PARAM['bfp']=self.PARAM['afn']*self.PARAM['bfn']/self.PARAM['afp']
        self.PARAM['cfp']=self.PARAM['afn']*self.PARAM['bfn']**2+self.PARAM['cfn']-self.PARAM['afp']*self.PARAM['bfp']**2
        self.PARAM['bgp']=self.PARAM['rg']-self.PARAM['agn']*(self.PARAM['rg']-self.PARAM['bgn'])/self.PARAM['agp']
        self.PARAM['cgp']=self.PARAM['agn']*(self.PARAM['rg']-self.PARAM['bgn'])**2+self.PARAM['cgn']-self.PARAM['agp']*(self.PARAM['rg']-self.PARAM['bgp'])**2
        
        self.PARAM['bhp']=self.PARAM['rh']-self.PARAM['ahn']*(self.PARAM['rh']-self.PARAM['bhn'])/self.PARAM['ahp']
        self.PARAM['chp']=self.PARAM['ahn']*(self.PARAM['rh']-self.PARAM['bhn'])**2+self.PARAM['chn']-self.PARAM['ahp']*(self.PARAM['rh']-self.PARAM['bhp'])**2

    # set coefficients Y
    def set_Y(self):
        f0=self.PARAM['dt']/self.PARAM['tau']*self.PARAM['phi']
        g0=self.PARAM['dt']/self.PARAM['tau']
        self.Y['v_vv_S']=(f0*self.PARAM['afn']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['v_vv_L']=(f0*self.PARAM['afp']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['v_v_S']=(f0*(-2)*self.PARAM['afn']*self.PARAM['bfn']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['v_v_L']=(f0*(-2)*self.PARAM['afp']*self.PARAM['bfp']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['v_c_S']=((f0*(self.PARAM['afn']*self.PARAM['bfn']*self.PARAM['bfn']+self.PARAM['cfn']+self.PARAM['I0'])*2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        self.Y['v_c_L']=((f0*(self.PARAM['afp']*self.PARAM['bfp']*self.PARAM['bfp']+self.PARAM['cfp']+self.PARAM['I0'])*2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        self.Y['v_n']=(-f0*(2**self.BIT_Y_SHIFT)).astype(int)
        self.Y['v_q']=(-f0*(2**self.BIT_Y_SHIFT)).astype(int)
        self.Y['v_I']=(f0*self.PARAM['k']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['n_vv_S']=(g0*self.PARAM['agn']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['n_vv_L']=(g0*self.PARAM['agp']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['n_v_S']=(g0*(-2)*self.PARAM['agn']*self.PARAM['bgn']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['n_v_L']=(g0*(-2)*self.PARAM['agp']*self.PARAM['bgp']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['n_n']=(-g0*(2**self.BIT_Y_SHIFT)).astype(int)
        self.Y['n_c_S']=((g0*(self.PARAM['agn']*self.PARAM['bgn']*self.PARAM['bgn']+self.PARAM['cgn'])*2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        self.Y['n_c_L']=((g0*(self.PARAM['agp']*self.PARAM['bgp']*self.PARAM['bgp']+self.PARAM['cgp'])*2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        self.Y['rg']=(self.PARAM['rg']*(2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        
        h0=self.PARAM['dt']/self.PARAM['tau']*self.PARAM['epsq']
        self.Y['q_vv_S']=(h0*self.PARAM['ahn']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['q_vv_L']=(h0*self.PARAM['ahp']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['q_v_S']=(h0*(-2)*self.PARAM['ahn']*self.PARAM['bhn']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['q_v_L']=(h0*(-2)*self.PARAM['ahp']*self.PARAM['bhp']*2**self.BIT_Y_SHIFT).astype(int)
        self.Y['q_q']=(-h0*(2**self.BIT_Y_SHIFT)).astype(int)
        self.Y['q_c_S']=((h0*(self.PARAM['ahn']*self.PARAM['bhn']*self.PARAM['bhn']+self.PARAM['chn'])*2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        self.Y['q_c_L']=((h0*(self.PARAM['ahp']*self.PARAM['bhp']*self.PARAM['bhp']+self.PARAM['chp'])*2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
        self.Y['rh']=(self.PARAM['rh']*(2**self.BIT_WIDTH_FRACTIONAL)).astype(int)
            

    # calculate Δv for the RSexhi, RSinhi, FS, LTS, IB, and EB modes
    def dv0(self,I_float):
        B=self.BIT_Y_SHIFT
        I=(I_float*2**self.BIT_WIDTH_FRACTIONAL).astype(int)
        v=self.state_variable_v
        n=self.state_variable_n
        q=self.state_variable_q
        vv = ((v * v) // (1 << self.BIT_WIDTH_FRACTIONAL)).astype(int)
        
        v0 = np.where(
            v < 0,
            (self.Y['v_vv_S'] * vv>>B)+(self.Y['v_v_S'] * v>>B)+self.Y['v_c_S']+(self.Y['v_n'] * n>>B)+(self.Y['v_q'] * q>>B)+(self.Y['v_I'] * I>>B),
            (self.Y['v_vv_L'] * vv>>B)+(self.Y['v_v_L'] * v>>B)+self.Y['v_c_L']+(self.Y['v_n'] * n>>B)+(self.Y['v_q'] * q>>B)+(self.Y['v_I'] * I>>B)
        )

        return v0

    # calculate Δn for the RSexhi, RSinhi, FS, EB and Class2 modes
    def dn0(self):
        B=self.BIT_Y_SHIFT
        v=self.state_variable_v
        n=self.state_variable_n
        vv=(v*v >> self.BIT_WIDTH_FRACTIONAL).astype(int)
        n0 = np.where(
            v<self.Y['rg'],
            (self.Y['n_vv_S']*vv >> B)+(self.Y['n_v_S']*v >> B)+self.Y['n_c_S']+(self.Y['n_n']*n >> B),
            (self.Y['n_vv_L']*vv >> B)+(self.Y['n_v_L']*v >> B)+self.Y['n_c_L']+(self.Y['n_n']*n >> B)
        )

        return n0

    # calculate Δq for the RSexhi, RSinhi, FS, LTS, IB, EB and PB modes
    def dq0(self):
        B=self.BIT_Y_SHIFT
        v=self.state_variable_v
        q=self.state_variable_q
        vv=(v*v >> self.BIT_WIDTH_FRACTIONAL).astype(int)
        q0 = np.where(
            v<self.Y['rh'],
            (self.Y['q_vv_S']*vv >> B)+(self.Y['q_v_S']*v >> B)+self.Y['q_c_S']+(self.Y['q_q']*q >> B),
            (self.Y['q_vv_L']*vv >> B)+(self.Y['q_v_L']*v >> B)+self.Y['q_c_L']+(self.Y['q_q']*q >> B)
        )
        return q0

    # update all state variables
    def update(self,I_float):
        dv=self.dv0(I_float)
        dn=self.dn0()
        dq=self.dq0()
        self.state_variable_v+=dv
        self.state_variable_n+=dn
        self.state_variable_q+=dq
        

    # get membrane potential
    def get_membrane_potential(self):
        return self.state_variable_v/2**self.BIT_WIDTH_FRACTIONAL

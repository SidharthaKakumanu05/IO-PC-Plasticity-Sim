import numpy as np

class IONeuron:
    def __init__(self, tau_v=20e-3, tau_u=200e-3, g_u=0.5,
                 v_rest=-65e-3, v_reset=-70e-3, v_th=-50e-3,
                 r_m=10e6, dt=1e-3):
        """
        IO neuron with resonate-and-fire dynamics and a slow recovery variable.
        """
        self.tau_v = tau_v
        self.tau_u = tau_u
        self.g_u = g_u
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.r_m = r_m
        self.dt = dt

        self.v = v_rest
        self.u = 0.0
        self.spikes = []

    def step(self, I, t, I_couple=0.0):
        dv = (-(self.v - self.v_rest) + self.r_m*(I + I_couple) + self.g_u*self.u) * (self.dt / self.tau_v)
        self.v += dv

        du = (-self.u + (self.v - self.v_rest)) * (self.dt / self.tau_u)
        self.u += du

        if self.v >= self.v_th:
            self.v = self.v_reset
            if len(self.spikes) == 0 or self.spikes[-1] != t:
                self.spikes.append(t)


class PurkinjeCell:
    def __init__(self, w_init=0.5):
        self.w = w_init
        self.history = []

    def apply_plasticity(self, pf_spike, cf_spikes,
                         ltd_window=0.1, null_window=0.2,
                         eta_ltd=0.01, eta_ltp=0.005):
        if len(cf_spikes) == 0:
            return
        delta_t = min([abs(pf_spike - cf) for cf in cf_spikes])

        if delta_t <= ltd_window:
            # LTD
            self.w -= eta_ltd * self.w
        elif delta_t > null_window:
            # LTP
            self.w += eta_ltp * (1 - self.w)

        self.w = np.clip(self.w, 0.0, 1.0)
        self.history.append(self.w)
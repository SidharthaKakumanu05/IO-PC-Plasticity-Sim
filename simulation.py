import numpy as np
from neurons import IONeuron, PurkinjeCell

def run_network(n_io=20, g_c=1e-9, tau_u=200e-3, g_u=0.5,
                I_tonic=1e-9, pf_rate=5.0, T=10.0, dt=1e-3):
    """
    Simulate a network of IO neurons coupled by gap junctions and projecting to a PC.
    """
    time = np.arange(0, T, dt)
    io_neurons = [IONeuron(dt=dt, tau_u=tau_u, g_u=g_u) for _ in range(n_io)]
    pc = PurkinjeCell()

    # PF spikes (Poisson)
    pf_spikes = time[np.random.rand(len(time)) < pf_rate*dt]
    io_voltages = [[] for _ in range(n_io)]
    io_spike_trains = [[] for _ in range(n_io)]
    cf_spikes = []

    for t in time:
        v_mean = np.mean([io.v for io in io_neurons])

        for i, io in enumerate(io_neurons):
            I_couple = g_c * (v_mean - io.v)
            io.step(I=I_tonic, t=t, I_couple=I_couple)
            io_voltages[i].append(io.v)

            if len(io.spikes) > 0 and io.spikes[-1] == t:
                io_spike_trains[i].append(t)
                cf_spikes.append(t)

        # Plasticity update for PF spikes near CF events
        for pf in pf_spikes:
            if abs(pf - t) < 0.01:
                pc.apply_plasticity(pf, cf_spikes)

    return time, io_voltages, io_spike_trains, pf_spikes, pc.history
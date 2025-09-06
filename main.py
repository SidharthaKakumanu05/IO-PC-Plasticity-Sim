from simulation import run_network, measure_frequency
from visualize import plot_io_activity, plot_pc_weights, plot_sweep

def sweep_parameter(param_values, param_name="tau_u", **kwargs):
    """
    Sweep a parameter and measure IO frequency + final PC weight.
    """
    results = {}
    for val in param_values:
        sim_args = kwargs.copy()
        sim_args[param_name] = val
        time, io_voltages, io_spike_trains, pf_spikes, weights = run_network(**sim_args)
        dt = time[1] - time[0]
        freq = measure_frequency(io_voltages[0], dt)
        results[val] = (freq, weights[-1] if weights else None)
    return results


if __name__ == "__main__":
    # Single run
    time, io_voltages, io_spike_trains, pf_spikes, weights = run_network(
        n_io=50, T=10.0, tau_u=200e-3, g_u=0.5, I_tonic=0.5e-9, pf_rate=5.0
    )

    plot_io_activity(time, io_voltages, io_spike_trains, pf_spikes, n_plot=5)
    plot_pc_weights(weights)

    # Sweep tau_u
    tau_vals = [100e-3, 200e-3, 400e-3, 800e-3]
    results = sweep_parameter(tau_vals, param_name="tau_u", n_io=20, T=10.0, I_tonic=0.5e-9)
    plot_sweep(results, param_name="tau_u")

    # Sweep I_tonic
    I_vals = [0.1e-9, 0.5e-9, 1e-9, 2e-9]
    results = sweep_parameter(I_vals, param_name="I_tonic", n_io=20, T=10.0, tau_u=200e-3)
    plot_sweep(results, param_name="I_tonic")

    # Sweep g_u
    g_vals = [0.2, 0.5, 0.8, 1.0]
    results = sweep_parameter(g_vals, param_name="g_u", n_io=20, T=10.0, tau_u=200e-3, I_tonic=0.5e-9)
    plot_sweep(results, param_name="g_u")
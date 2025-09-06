from simulation import run_network
from visualize import plot_io_activity, plot_pc_weights

if __name__ == "__main__":
    # Run with many IO neurons
    time, io_voltages, io_spike_trains, pf_spikes, weights = run_network(
        n_io=50, T=10.0, tau_u=200e-3, g_u=0.5, I_tonic=1e-9, pf_rate=5.0
    )

    # Plot results
    plot_io_activity(time, io_voltages, io_spike_trains, pf_spikes, n_plot=5)
    plot_pc_weights(weights)
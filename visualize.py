import matplotlib.pyplot as plt
import numpy as np

def plot_io_activity(time, io_voltages, io_spike_trains, pf_spikes, n_plot=5):
    """
    Plot IO voltage traces and raster plot of IO (CF) and PF spikes.
    """
    plt.figure(figsize=(10,7))

    # Voltage of first few IO neurons
    for i in range(min(n_plot, len(io_voltages))):
        plt.plot(time, io_voltages[i], label=f"IO {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (a.u.)")
    plt.title("IO Neuron Membrane Potentials")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Raster plot
    plt.figure(figsize=(10,6))
    for i, spikes in enumerate(io_spike_trains):
        plt.scatter(spikes, [i]*len(spikes), marker='|', color='red')
    plt.scatter(pf_spikes, [-1]*len(pf_spikes), marker='|', color='blue')
    plt.yticks([-1] + list(range(len(io_spike_trains))), ["PF"] + [f"IO {i}" for i in range(len(io_spike_trains))])
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron")
    plt.title("Spike Raster: IOs (red) and PFs (blue)")
    plt.tight_layout()
    plt.show()


def plot_pc_weights(weights):
    plt.figure(figsize=(7,4))
    plt.plot(weights, color="green")
    plt.xlabel("Update step")
    plt.ylabel("Synaptic Weight")
    plt.title("Purkinje Synaptic Weight Evolution (PF→PC)")
    plt.tight_layout()
    plt.show()
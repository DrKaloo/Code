import numpy as np
import matplotlib.pyplot as plt

# Neuron parameters
R = 10e6            # Membrane resistance: 10 MΩ
C = 0.1e-9          # Membrane capacitance: 0.1 nF
tau = R * C         # Time constant: tau = RC = 1 ms
E = -75e-3          # Resting potential: -75 mV
V_thresh = -50e-3   # Spike threshold: -50 mV
V_reset = -75e-3    # Reset potential: -75 mV

# Simulation parameters
dt = 0.1e-3                     # Time step: 0.1 ms
T = 200e-3                      # Total time: 200 ms
t = np.arange(0, T, dt)

def simulate_lif(I_app, t, dt):
    """
    Simulate LIF neuron with applied current using Euler integration.

    Parameters:
    -----------
    I_app : float or array
        Applied current (A)
    t : array
        Time vector (s)
    dt : float
        Time step (s)

    Returns:
    --------
    V : array
        Membrane potential over time (V)
    spikes : list
        Spike times (s)
    """

    V = np.zeros(len(t))
    V[0] = E
    spikes = []

    # I_app -> array if scalar
    if np.isscalar(I_app):
        I_app = np.ones(len(t)) * I_app

    # Euler integration: V(t+dt) = V(t) + dt * dV/dt
    for i in range(1, len(t)):
        # LIF equation: dV/dt = (E - V)/tau + I/C
        dV = dt * ((E - V[i-1])/tau + I_app[i]/C)
        V[i] = V[i-1] + dV

        # Checks for threshold crossing
        if V[i] >= V_thresh:
            spikes.append(t[i])
            V[i] = V_reset  # Reset after spike

    return V, spikes


def compute_fi_curve(I_range, duration=1.0):
    """
    Compute frequency-current (F-I) curve.

    Parameters:
    -----------
    I_range : array
        Range of input currents to test (A)
    duration : float
        Duration of each simulation (s)

    Returns:
    --------
    frequencies : array
        Firing rates (Hz) for each current level
    """

    t_fi = np.arange(0, duration, dt)
    frequencies = []

    for I in I_range:
        V, spikes = simulate_lif(I, t_fi, dt)

        # Count spikes after initial transient (first 100 ms)
        stable_spikes = [s for s in spikes if s > 0.1]
        freq = len(stable_spikes) / (duration - 0.1)  # spikes/second
        frequencies.append(freq)

    return np.array(frequencies)


# Analysis 1: Response to step current

print("LIF Neuron Simulation")

print(f"Parameters:")
print(f"  R = {R*1e-6:.1f} MΩ")
print(f"  C = {C*1e9:.1f} nF")
print(f"  tau = {tau*1e3:.1f} ms")
print(f"  E_rest = {E*1e3:.1f} mV")
print(f"  V_thresh = {V_thresh*1e3:.1f} mV")


fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Panel 1: Step current response
ax1 = axes[0, 0]
I_step = np.zeros(len(t))
I_step[int(50e-3/dt):] = 3.0e-9  # 3.0 nA step at 50 ms

V, spikes = simulate_lif(I_step, t, dt)

ax1.plot(t*1e3, V*1e3, 'b-', linewidth=1.5, label='Membrane potential')
ax1.axhline(V_thresh*1e3, color='r', linestyle='--', alpha=0.7, label='Threshold')
for spike in spikes:
    ax1.axvline(spike*1e3, color='gray', alpha=0.3, linewidth=0.8)
ax1.set_ylabel('Voltage (mV)')
ax1.set_title('Response to Step Current (I = 3.0 nA)')
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

print(f"\n Step current (3.0 nA):")
print(f"   Number of spikes: {len(spikes)}")
if len(spikes) > 1:
    isi_mean = np.mean(np.diff(spikes)) * 1e3
    print(f"   Mean ISI: {isi_mean:.1f} ms")
    print(f"   Firing rate: {1000/isi_mean:.1f} Hz")

# Panel 2: Input current
ax2 = axes[1, 0]
ax2.plot(t*1e3, I_step*1e9, 'k-', linewidth=2)
ax2.set_ylabel('Current (nA)')
ax2.set_xlabel('Time (ms)')
ax2.set_title('Applied Current')
ax2.grid(alpha=0.3)

# Panel 3: Multiple current levels
ax3 = axes[0, 1]
currents = [0.5e-9, 1.0e-9, 1.5e-9, 2.0e-9]
colors = plt.cm.viridis(np.linspace(0, 1, len(currents)))

for I_level, color in zip(currents, colors):
    V, spikes = simulate_lif(I_level, t, dt)
    ax3.plot(t*1e3, V*1e3, color=color, linewidth=1.5,
             label=f'{I_level*1e9:.1f} nA', alpha=0.8)

ax3.axhline(V_thresh*1e3, color='r', linestyle='--', alpha=0.5, linewidth=2)
ax3.set_ylabel('Voltage (mV)')
ax3.set_title('Response to Different Current Levels')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(alpha=0.3)

# Panel 4: Inter-spike interval distribution
ax4 = axes[1, 1]
t_long = np.arange(0, 1.0, dt)
V_long, spikes_long = simulate_lif(3.0e-9, t_long, dt)

if len(spikes_long) > 2:
    isis = np.diff(spikes_long) * 1e3  # Convert to ms
    ax4.hist(isis, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(isis), color='r', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(isis):.1f} ms')
    ax4.set_xlabel('Inter-Spike Interval (ms)')
    ax4.set_ylabel('Count')
    ax4.set_title('ISI Distribution (I = 3.0 nA, 1 sec)')
    ax4.legend()
    ax4.grid(alpha=0.3)

# Panel 5: F-I Curve
ax5 = axes[2, 0]
I_range = np.linspace(0, 3e-9, 25)
frequencies = compute_fi_curve(I_range, duration=1.0)

ax5.plot(I_range*1e9, frequencies, 'o-', linewidth=2, markersize=7,
         color='steelblue', markerfacecolor='steelblue', alpha=0.8)
ax5.set_xlabel('Applied Current (nA)')
ax5.set_ylabel('Firing Rate (Hz)')
ax5.set_title('Frequency-Current (F-I) Curve')
ax5.grid(alpha=0.3)

# Find rheobase (minimum A to fire)
firing_mask = frequencies > 0
if np.any(firing_mask):
    rheobase = I_range[firing_mask][0]
    print(f"\nRheobase (minimum firing current): {rheobase*1e9:.2f} nA")

# Panel 6: Raster plot
ax6 = axes[2, 1]
currents_raster = np.linspace(0.5e-9, 2.5e-9, 12)
t_raster = np.arange(0, 0.5, dt)

for idx, I_level in enumerate(currents_raster):
    V, spikes = simulate_lif(I_level, t_raster, dt)
    if spikes:
        ax6.scatter(np.array(spikes)*1e3, [I_level*1e9]*len(spikes),
                   marker='|', s=100, color='black', alpha=0.7)

ax6.set_xlabel('Time (ms)')
ax6.set_ylabel('Applied Current (nA)')
ax6.set_title('Raster Plot: Spiking vs Current')
ax6.grid(alpha=0.3)

plt.tight_layout()

plt.show()

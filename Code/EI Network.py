import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import networkx as nx

# ---------------------------------
# NETWORK PARAMETERS


# Visualisation network size
N_exc = 80   # Excitatory neurons
N_inh = 20   # Inhibitory neurons
N_total = N_exc + N_inh

# Population dynamics parameters
tau_E = 3.5e-3  # Excitatory time constant (3.5 ms)
tau_I = 1.0e-3  # Inhibitory time constant (1.0 ms, faster)

# Coupling weights
w_EE = 0.8      # E → E (recurrent excitation)
w_EI = 1.2      # E → I (drive inhibition)
w_IE = -1.8     # I → E (inhibition, negative)

# Simulation parameters
dt = 0.1e-3     # Time step (0.1 ms)
n_steps = 1000  # Total steps (100 ms)

# Connection probability for visualisation
p_conn = 0.18   # 18% connectivity


print("Excitatory-Inhibitory Neural Network Simulation")

print(f"\nNetwork: {N_exc} excitatory + {N_inh} inhibitory neurons")
print(f"Time constants: tau_E = {tau_E*1000:.1f} ms, tau_I = {tau_I*1000:.1f} ms")
print(f"Simulation: {n_steps * dt * 1000:.1f} ms total")
print("--------------")

# ------------------------------
# Building Network Structure


np.random.seed(42)

# Create spatial layout
pos = {}

# Excitatory neurons (outer ring)
exc_angles = np.linspace(0, 2*np.pi, N_exc, endpoint=False)
for i in range(N_exc):
    pos[i] = (1.5 * np.cos(exc_angles[i]), 1.5 * np.sin(exc_angles[i]))

# Inhibitory neurons (inner ring)
inh_angles = np.linspace(0, 2*np.pi, N_inh, endpoint=False)
for i in range(N_inh):
    pos[N_exc + i] = (0.7 * np.cos(inh_angles[i]), 0.7 * np.sin(inh_angles[i]))

# Neuron type labels
neuron_type = ['E'] * N_exc + ['I'] * N_inh

# Connectivity graph
graph = nx.DiGraph()
graph.add_nodes_from(range(N_total))

for i in range(N_total):
    for j in range(N_total):
        if i != j and np.random.random() < p_conn:
            graph.add_edge(i, j)

print(f"\nNetwork structure:")
print(f" Total connections: {graph.number_of_edges()}")
print(f" Average in-degree: {graph.number_of_edges() / N_total:.1f}")

# -----------------------------------
# Simulate Population Dynamics


print(f"\n Simulating population dynamics:")

# Initialise population activities
n_E = np.zeros(n_steps)  # Excitatory population
n_I = np.zeros(n_steps)  # Inhibitory population

# Initial conditions (start with some activity nA)
n_E[0] = 1.5e-9  # 1.5 nA
n_I[0] = 0.5e-9  # 0.5 nA

# External input pattern (pulsed input to excitatory population)
I_ext = np.zeros(n_steps)
I_ext[0:100] = 3.0e-9       # First pulse (0-10 ms)
I_ext[200:280] = 2.5e-9     # Second pulse (20-28 ms)
I_ext[400:480] = 2.0e-9     # Third pulse (40-48 ms)
I_ext[600:680] = 2.5e-9     # Fourth pulse (60-68 ms)
I_ext[800:880] = 2.0e-9     # Fifth pulse (80-88 ms)

# Euler integration (2-population rate model)
for t in range(1, n_steps):
    # Excitatory population dynamics
    dn_E = (dt / tau_E) * (-n_E[t-1] + w_EE * n_E[t-1] + w_IE * n_I[t-1] + I_ext[t])
    n_E[t] = n_E[t-1] + dn_E

    # Inhibitory population dynamics
    dn_I = (dt / tau_I) * (-n_I[t-1] + w_EI * n_E[t])
    n_I[t] = n_I[t-1] + dn_I

    # Apply physiological bounds
    n_E[t] = np.clip(n_E[t], 0, 10e-9)
    n_I[t] = np.clip(n_I[t], 0, 10e-9)

print(f"  Excitatory range: [{np.min(n_E)*1e9:.2f}, {np.max(n_E)*1e9:.2f}] nA")
print(f"  Inhibitory range: [{np.min(n_I)*1e9:.2f}, {np.max(n_I)*1e9:.2f}] nA")

# ----------------------------------------------------------------------
# Convert To Individual Neuron Activities


print(f"\n Generating individual neuron activities:")

# Sample individual neurons from population distributions
r_neurons = np.zeros((n_steps, N_total))

for t in range(n_steps):
    # Excitatory neurons sample from E population
    for i in range(N_exc):
        noise = np.random.normal(0, 0.2e-9)
        r_neurons[t, i] = max(0, n_E[t] + noise)

    # Inhibitory neurons sample from I population
    for i in range(N_exc, N_total):
        noise = np.random.normal(0, 0.2e-9)
        r_neurons[t, i] = max(0, n_I[t] + noise)

# ----------------------------
# Animation

print(f"\n Creating animation:")

fig = plt.figure(figsize=(16, 8))
fig.suptitle('Excitatory-Inhibitory Neural Network', fontsize=14, fontweight='bold')
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])

# Create subplots
ax_network = fig.add_subplot(gs[0, :])
ax_trace = fig.add_subplot(gs[1, 0])
ax_input = fig.add_subplot(gs[1, 1])

# Time axis (in ms)
time_ms = np.arange(n_steps) * dt * 1000

# Separate edges by source neuron type
exc_edges = [(i, j) for i, j in graph.edges() if i < N_exc]
inh_edges = [(i, j) for i, j in graph.edges() if i >= N_exc]

def animate(frame):
    """Animation function for each frame"""
    ax_network.clear()
    ax_trace.clear()
    ax_input.clear()

    # Current neural activities
    r_current = r_neurons[frame, :]
    r_norm = r_current / 5e-9  # Normalize to 0-1 for visualization

    # Count active neurons (threshold at 15%)
    n_active = np.sum(r_norm > 0.15)

    # ---------------------------
    # Network Visualisation

    ax_network.set_title(
        f't = {time_ms[frame]:.1f} ms | Active: {n_active}/{N_total} | ' +
        f'E = {n_E[frame]*1e9:.2f} nA, I = {n_I[frame]*1e9:.2f} nA',
        fontsize=12, fontweight='bold', pad=10
    )

    # Draw connections (excitatory - blue, inhibitory - red)
    nx.draw_networkx_edges(graph, pos, exc_edges, ax=ax_network,
                          alpha=0.20, edge_color='steelblue', width=0.5,
                          arrows=True, arrowsize=4, arrowstyle='->')

    nx.draw_networkx_edges(graph, pos, inh_edges, ax=ax_network,
                          alpha=0.35, edge_color='crimson', width=0.8,
                          arrows=True, arrowsize=6, arrowstyle='->')

    # Draw neurons with activity-dependent colors and sizes
    colors = []
    sizes = []

    for i in range(N_total):
        act = r_norm[i]

        if i < N_exc:  # Excitatory neuron
            if act < 0.1:
                colors.append((0.0, 0.0, 0.2))  # Dark blue (quiet)
            else:
                # Cyan gradient (active)
                colors.append((0.0, 0.2 + 0.8*act, 0.2 + 0.8*act))
        else:  # Inhibitory neuron
            if act < 0.1:
                colors.append((0.2, 0.0, 0.0))  # Dark red (quiet)
            else:
                # Red gradient (active)
                colors.append((0.2 + 0.8*act, 0.0, 0.0))

        # Size scales with activity
        sizes.append(50 + 400*act)

    nx.draw_networkx_nodes(graph, pos, ax=ax_network,
                          node_color=colors, node_size=sizes,
                          alpha=0.9, edgecolors='white', linewidths=1.5)

    ax_network.axis('off')
    ax_network.set_aspect('equal')

    # -------------------------------
    # Population Activity Traces

    ax_trace.set_title('Population Activity', fontsize=11, fontweight='bold')

    # Plot traces up to current frame
    ax_trace.plot(time_ms[:frame+1], n_E[:frame+1]*1e9,
                 'b-', linewidth=2.5, label='Excitatory', alpha=0.9)
    ax_trace.plot(time_ms[:frame+1], n_I[:frame+1]*1e9,
                 'r-', linewidth=2.5, label='Inhibitory', alpha=0.9)

    # Current time indicator
    ax_trace.axvline(time_ms[frame], color='black',
                    linestyle='--', alpha=0.4, linewidth=1)

    # Current values
    ax_trace.plot(time_ms[frame], n_E[frame]*1e9,
                 'o', color='blue', markersize=10)
    ax_trace.plot(time_ms[frame], n_I[frame]*1e9,
                 'o', color='red', markersize=10)

    ax_trace.set_xlabel('Time (ms)', fontsize=10)
    ax_trace.set_ylabel('Activity (nA)', fontsize=10)
    ax_trace.set_xlim([max(0, time_ms[frame]-20), time_ms[frame]+5])
    ax_trace.set_ylim([0, 4])
    ax_trace.legend(loc='upper right')
    ax_trace.grid(alpha=0.3)

    # --------------------
    # External Input

    ax_input.set_title('External Input', fontsize=11, fontweight='bold')

    # Plot input up to current frame
    ax_input.plot(time_ms[:frame+1], I_ext[:frame+1]*1e9,
                 'orange', linewidth=2.5, alpha=0.9)

    # Current time indicator
    ax_input.axvline(time_ms[frame], color='black',
                    linestyle='--', alpha=0.4, linewidth=1)

    # Current value
    ax_input.plot(time_ms[frame], I_ext[frame]*1e9,
                 'o', color='orange', markersize=10)

    ax_input.set_xlabel('Time (ms)', fontsize=10)
    ax_input.set_ylabel('Input (nA)', fontsize=10)
    ax_input.set_xlim([max(0, time_ms[frame]-20), time_ms[frame]+5])
    ax_input.grid(alpha=0.3)

    return []

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=n_steps,
                              interval=20, blit=False, repeat=True)

plt.tight_layout()

print("Animation: :)")

print("\n Showing animation window:")

print("Exit to close")

plt.show()
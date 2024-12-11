import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

# Simulation parameters
room_dim = [5, 4, 3]  # Room dimensions in meters
mic_positions = np.array(
    [[2.5, 2.5, 1.5], [2, 2, 1.5]]  # Microphone x-coordinate  # Microphone y-coordinate
)  # Centered microphone array
source_pos = [3, 3, 1.5]  # Speaker position
noise_pos = [1, 1, 1.5]  # Noise source position

# Create a room with pyroomacoustics
room = pra.ShoeBox(room_dim, fs=16000, max_order=1, absorption=0.4)

# Add microphone array
mics = pra.MicrophoneArray(
    mic_positions.T, room.fs
)  # Transpose to ensure correct shape
room.add_microphone_array(mics)

# Add sources to the room
room.add_source(source_pos)
room.add_source(noise_pos)

# Simulate a simple signal for the microphone array
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(room.fs * duration))
signal = np.sin(2 * np.pi * 440 * t)  # A 440 Hz sine wave

# Add the signal to the microphone array
mics.signals = np.tile(signal, (mics.M, 1))  # M is the number of microphones


def compute_srp(room, grid_points):
    srp_map = np.zeros(grid_points.shape[0])
    for idx, point in enumerate(grid_points):
        delays = room.sources[0].distance(point) / 343.0  # Speed of sound = 343 m/s
        signal = room.mic_array.signals[
            0
        ]  # Example signal, should simulate for sources
        srp_map[idx] = np.sum(np.correlate(signal, signal, mode="valid"))
    return srp_map


# Compute SRP (Steered Response Power) map
X = np.linspace(0, room_dim[0], 100)  # X range
Y = np.linspace(0, room_dim[1], 100)  # Y range
X, Y = np.meshgrid(X, Y)
grid_points = np.c_[
    X.ravel(), Y.ravel(), np.full(X.ravel().shape, 1.5)
]  # Assume height = 1.5 m

srp_map = compute_srp(room, grid_points)
Z = srp_map.reshape(X.shape)

# Plot the SRP map
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Power")
plt.title("SRP Map")
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

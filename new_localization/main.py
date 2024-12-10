import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from xsrp_project.xsrp.conventional_srp import ConventionalSrp

# Define the microphone positions
# Ensure channels are in the correct order
print(
    "Please enter the microphone positions in the format [x1, y1], [x2, y2], [x3, y3], [x4, y4]"
)
mic_positions = np.array(
    [
        [0.0000, 0.4500],
        [0.4500, 0.0000],
        [0.0000, -0.4500],
        [-0.4500, 0.0000],
    ]
)

fs = 44100  # Sampling rate
frame_size = 1024
hop_size = 512  # Overlap may be used, depending on your STFT
channels = 4  # Number of mics

# Initialize the ConventionalSrp object
srp_func = ConventionalSrp(
    fs,
    grid_type="doa_2D",
    n_grid_cells=200,
    mic_positions=mic_positions,
    interpolation=False,
    mode="gcc_phat_freq",
    n_average_samples=5,
    freq_cutoff_in_hz=None,
)

# Set up a matplotlib figure for live updates
fig = plt.figure()

# Create a 2D scatter plot
ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_title("Real-time Sound Localization")
scat = ax1.scatter([], [], c="red", s=50, label="Estimated Source")
ax1.legend()

# Create a 3D plot
ax2 = fig.add_subplot(111, projection="3d")  # 1 row, 2 columns, 2nd subplot
ax2.set_title("3D Data Map")

# A buffer to store audio data
audio_buffer = np.zeros((channels, frame_size))


# This callback will be called by sounddevice for each block of audio data
def audio_callback(indata, frames, time, status):
    # indata: shape (frames, channels)
    global audio_buffer, srp_func, scat

    # Move data into buffer (for simplicity, assume frames == frame_size)
    audio_buffer = indata.T  # shape: (channels, frame_size)

    # Process with ConventionalSrp
    # forward expects signals shape: (n_mics, n_samples)
    # If you need STFT, do it here. For simplicity, let's assume direct time-domain input.
    estimated_positions, srp_map, candidate_grid = srp_func.forward(audio_buffer)

    # estimated_positions might have one or more sources; take the first source if present
    if estimated_positions is not None and len(estimated_positions) > 0:
        est_pos = estimated_positions[0]
        # Update the scatter plot with the estimated position
        scat.set_offsets([est_pos[0], est_pos[1]])

        # I want to plot the SRP map
        x = candidate_grid[:, 0].reshape((200, 200))
        y = candidate_grid[:, 1].reshape((200, 200))
        z = srp_map.reshape((200, 200))

        ax2.clear()
        ax2.plot_surface(x, y, z, cmap="viridis", edgecolor="none")
        ax2.set_title("SRP Map")
        ax2.set_xlabel("x (cm)")
        ax2.set_ylabel("y (cm)")
        ax2.set_zlabel("Power")

    # Force matplotlib to redraw
    plt.pause(0.001)


# Print out connected audio devices
print(sd.query_devices(), "\n")

device_index = input("Please enter the device index: ")

# Configure the input stream
stream = sd.InputStream(
    device=int(device_index),  # Please specify the device index
    channels=channels,
    samplerate=fs,
    blocksize=frame_size,
    callback=audio_callback,
)

# Start the stream
with stream:
    print("Press Ctrl+C to stop")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import noisereduce as nr
from xsrp_project.xsrp.conventional_srp import ConventionalSrp
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# Define the microphone positions
# Ensure channels are in the correct order
print(
    "Please enter the microphone positions in the format [x1, y1], [x2, y2], [x3, y3], [x4, y4]"
)
mic_positions = np.array(
    [
        [0.0000, 0.4500],
        [-0.4500, 0.0000],
        [0.0000, -0.4500],
        [0.4500, 0.0000],
    ]
)

print(f"Microphone positions shape: {mic_positions.shape}")

fs = 44100  # Sampling rate
# frame_size = 1024
frame_size = 512
hop_size = 512  # Overlap may be used, depending on your STFT
channels = 4  # Number of mics

# Initialize the ConventionalSrp object
srp_func = ConventionalSrp(
    fs,
    grid_type="2D",
    n_grid_cells=50,
    room_dims=[10, 10],
    mic_positions=mic_positions,
    interpolation=False,
    mode="gcc_phat_freq",
    n_average_samples=5,
    freq_cutoff_low_in_hz=None,
    freq_cutoff_high_in_hz=None,
)

# Set up a matplotlib figure for live updates
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_title("Real-time Sound Localization")
ax1.scatter(
    mic_positions[:, 0],
    mic_positions[:, 1],
    c="blue",
    s=50,
    label="Microphone Positions",
)
scat = ax1.scatter([], [], c="red", marker="*", s=50, label="Estimated Source")
ax1.legend()

ax2 = fig.add_subplot(111, projection="3d")
ax2.set_title("SRC Map")
ax2.set_xlabel("x (cm)")
ax2.set_ylabel("y (cm)")
ax2.set_zlabel("power")


# A buffer to store audio data
audio_buffer = np.zeros((channels, frame_size))


# This callback will be called by sounddevice for each block of audio data
def audio_callback(indata, frames, time, status):
    # indata: shape (frames, channels)
    global audio_buffer, srp_func, scat

    audio_buffer = indata.T  # shape: (channels, frame_size)
    audio_buffer = butter_bandpass_filter(audio_buffer, 90, 200, fs)
    audio_buffer = nr.reduce_noise(y=audio_buffer, y_noise=audio_buffer, sr=fs)


def update_plot(frame):
    global audio_buffer, srp_func, scat

    estimated_positions, srp_map, candidate_grid = srp_func.forward(audio_buffer)
    print(f"SRP shape: {srp_map.shape}")
    print(f"Candidate grid shape: {candidate_grid.asarray().shape}")
    print(f"Candidate grid X: {candidate_grid.asarray()[:, 0]}")

    # estimated_positions might have one or more sources; take the first source if present
    if estimated_positions is not None and len(estimated_positions) > 0:
        num_cells = int(np.sqrt(candidate_grid.asarray().shape[0]))
        assert (
            srp_map.size == num_cells**2
        ), "SRP map size does not match grid dimensions."

        # Reshape SRP map and grid coordinates
        srp_map = srp_map.reshape((num_cells, num_cells))
        candidate_grid_x = candidate_grid.asarray()[:, 0].reshape(
            (num_cells, num_cells)
        )
        candidate_grid_y = candidate_grid.asarray()[:, 1].reshape(
            (num_cells, num_cells)
        )

        ax2.plot_surface(
            candidate_grid_x,
            candidate_grid_y,
            srp_map,
            cmap="jet",
            edgecolor="none",
        )
        scat.set_offsets([estimated_positions[0], estimated_positions[1]])

    return (scat,)


ani = animation.FuncAnimation(fig, update_plot, frames=range(100), blit=True)


# Print out connected audio devices
print(sd.query_devices(), "\n")

# device_index = input("Please enter the device index: ")
device_index = 13

# Configure the input stream
stream = sd.InputStream(
    device=int(device_index),  # Please specify the device index
    channels=channels,
    samplerate=fs,
    blocksize=frame_size,
    callback=audio_callback,
)

# Get multiple device indices from the user
# device_indices = input("Please enter the device indices (comma-separated): ")

# # Split the input by comma and convert each index to an integer
# device_indices = [int(index) for index in device_indices.split(',')]

# print(f"Device indices: {device_indices}")

# Create a list to store the InputStream instances
streams = []

# Start the stream
with stream:
    print("Press Ctrl+C to stop")
    plt.show()

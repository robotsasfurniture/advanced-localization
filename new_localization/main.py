from itertools import combinations
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import minimize
import pyroomacoustics as pra

# Import the GCC and NGCCPHAT classes from your model.py
from model import GCC, NGCCPHAT

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

###############################################
# Load the sample speech signal
###############################################
filename = "speech_small.wav"  # Adjust if needed
fs, signal = wavfile.read(filename)
signal = np.array(signal, dtype=float)

plt.figure()
plt.plot(signal)
plt.title("Original Speech Signal")
plt.show()

###############################################
# Create a 2D virtual room with microphones and a sound source
###############################################
room_dim = [3.0, 2.5]  # 2D: [length, width]
t60 = 0.6  # Reverberation time
snr = 10  # SNR in dB

# Microphone array (4 mics) in 2D: shape (2, 4)
mic_locs = np.array(
    [
        [1.5, 1.25 + 0.4500],
        [1.5 + 0.4500, 1.25],
        [1.5, 1.25 - 0.4500],
        [1.5 - 0.4500, 1.25],
    ]
).T

# Source location in 2D
source_loc = np.array([3, 1.20])

# Compute room absorption and max reflection order
e_absorption, max_order = pra.inverse_sabine(t60, room_dim)

# Create the shoebox room (2D)
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
)
c = room.c  # speed of sound

# Add source and mics
room.add_source(source_loc, signal=signal)
room.add_microphone(mic_locs)

###############################################
# Visualize the 2D layout of microphones and source
###############################################
plt.figure()
plt.scatter(mic_locs[0], mic_locs[1], c="b", label="microphones")
plt.scatter(source_loc[0], source_loc[1], c="r", label="sound source")
plt.xlim([0, room_dim[0]])
plt.ylim([0, room_dim[1]])
plt.title("Top View: Mic and Source Layout")
plt.legend()
plt.show()

###############################################
# Compute the true TDOAs for reference
###############################################
pairs_list = list(combinations(range(4), 2))
true_delays = []
for pairs in pairs_list:
    d = np.sqrt(np.sum((mic_locs[:, pairs[0]] - source_loc) ** 2)) - np.sqrt(
        np.sum((mic_locs[:, pairs[1]] - source_loc) ** 2)
    )
    true_delays.append(d * fs / c)
print("The true TDOAs are:", true_delays)

###############################################
# Simulate room sound propagation in 2D
###############################################
room.simulate(snr=snr)
y = room.mic_array.signals

# Plot the received signal at mic 0
plt.figure()
plt.plot(y[0])
plt.title("Signal received at mic 0 (Full)")
plt.show()

###############################################
# Select a segment of the received signal
###############################################
sig_len = 2048
start_idx = 20000
end_idx = start_idx + sig_len
x = torch.Tensor(y[:, start_idx:end_idx])

plt.figure()
plt.plot(range(start_idx, end_idx), x[0].squeeze().numpy())
plt.title("Signal segment at mic 0")
plt.show()

###############################################
# Load the GCC-PHAT and NGCC-PHAT modules
###############################################
max_tau = 23
gcc = GCC(max_tau)

# Initialize NGCC-PHAT model
ngcc = NGCCPHAT(max_tau, "classifier", True, sig_len, 128, fs)
ngcc.load_state_dict(
    torch.load("experiments/ngccphat/model.pth", map_location=torch.device("cpu"))
)
ngcc.eval()

###############################################
# Compute cross-correlations and TDOA estimates
###############################################
gcc_delays = []
ngcc_delays = []

for i, pairs in enumerate(pairs_list):
    x1 = x[pairs[0]].unsqueeze(0)  # shape: (1, seg_len)
    x2 = x[pairs[1]].unsqueeze(0)  # shape: (1, seg_len)

    with torch.no_grad():
        cc = gcc(x1, x2).squeeze()  # GCC-PHAT
        cc = cc / torch.max(cc)
        p = ngcc(x1, x2).squeeze()  # NGCC-PHAT probabilities
        p = p / torch.max(p)

    inds = np.arange(-max_tau, max_tau + 1)

    # Plot correlation functions
    plt.figure()
    plt.plot(inds, cc.numpy(), label="gcc-phat")
    plt.plot(inds, p.numpy(), label="ngcc-phat")
    plt.legend()

    shift_gcc = float(torch.argmax(cc)) - max_tau
    shift_ngcc = float(torch.argmax(p)) - max_tau
    gcc_delays.append(shift_gcc)
    ngcc_delays.append(shift_ngcc)

    plt.scatter(shift_gcc, 1.0, marker="*", c="blue")
    plt.scatter(shift_ngcc, 1.0, marker="*", c="green")
    plt.title(f"Correlation between mic {pairs[0]} and {pairs[1]}")
    plt.show()

    print("True TDOA (in samples):", true_delays[i])
    print("GCC-PHAT estimate:", shift_gcc)
    print("NGCC-PHAT estimate:", shift_ngcc)


###############################################
# Multilateration to estimate source position (2D)
###############################################
def loss(x, mic_locs, tdoas):
    # x is a 2D position [x, y]
    return sum(
        [
            (
                np.linalg.norm(x - mic_locs[:, pairs[0]])
                - np.linalg.norm(x - mic_locs[:, pairs[1]])
                - (tdoas[i] / fs * c)
            )
            ** 2
            for i, pairs in enumerate(pairs_list)
        ]
    )


guess = [0, 0]
bounds = ((0, room_dim[0]), (0, room_dim[1]))

xhat_gcc = minimize(loss, guess, args=(mic_locs[:2], gcc_delays), bounds=bounds).x
xhat_ngcc = minimize(loss, guess, args=(mic_locs[:2], ngcc_delays), bounds=bounds).x

print("Ground truth position:", source_loc[:2])
print("GCC estimate:", xhat_gcc)
print("NGCC estimate:", xhat_ngcc)

###############################################
# Visualize hyperbolas of TDOA constraints in 2D
###############################################
xx = np.linspace(0, room_dim[0], 100)
yy = np.linspace(0, room_dim[1], 100)
xx, yy = np.meshgrid(xx, yy)


def plot_hyperbolas(tdoas, name, estimate=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, pairs in enumerate(pairs_list):
        contour_data = (
            np.sqrt(
                (xx - mic_locs[0, pairs[0]]) ** 2 + (yy - mic_locs[1, pairs[0]]) ** 2
            )
            - np.sqrt(
                (xx - mic_locs[0, pairs[1]]) ** 2 + (yy - mic_locs[1, pairs[1]]) ** 2
            )
            - tdoas[i] / fs * c
        )
        plt.contour(xx, yy, contour_data, [0])
    ax.scatter(mic_locs[0], mic_locs[1], c="b", label="microphones")
    ax.scatter(source_loc[0], source_loc[1], c="r", label="sound source")
    if estimate is not None:
        ax.scatter(estimate[0], estimate[1], c="g", label="estimate", marker="*", s=200)
    ax.set_xlim([0, room_dim[0]])
    ax.set_ylim([0, room_dim[1]])
    plt.title(name)
    plt.legend()
    plt.show()


plot_hyperbolas(true_delays, "Ground Truth")
plot_hyperbolas(gcc_delays, "GCC-PHAT", xhat_gcc)
plot_hyperbolas(ngcc_delays, "NGCC-PHAT", xhat_ngcc)

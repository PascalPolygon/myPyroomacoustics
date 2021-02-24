import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist
from scipy.io import wavfile


def split_audio(audio):
    left = []
    right = []
    for i in range(0, len(audio) // 2, 2):
        left.append(audio[i * 2])
        left.append(audio[i * 2 + 1])
        right.append(audio[i * 2 + 2])
        right.append(audio[i * 2 + 3])
    left = np.array(left)
    right = np.array(right)
    return left, right
######
# We define a meaningful distance measure on the circle


# Location of original source
# azimuth = 61.0 / 180.0 * np.pi  # 60 degrees
azimuth = 61.0 / 180.0 * np.pi  # 60 degrees
distance = 0.5  # 3 meters

#######################
# algorithms parameters
SNR = 0.0  # signal-to-noise ratio
c = 343.0  # speed of sound
fs = 44100  # sampling frequency
# nfft = 256  # FFT size
nfft = 512
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# fs, audio = wavfile.read("examples/samples/guitar_16k.wav")

fs, audio = wavfile.read(
    "/home/pascal/EdgeAnalyticsPhase3/cloud/sample_audio/example_short.wav")

# compute the noise variance
sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2

# Create an anechoic room
# room_dim = np.r_[10.0, 10.0]
# room_dim = np.r_[5.2, 7.4]

# f2m = 1/3.2808


def f2m(x):
    return x*(1/3.2808)


def m2f(x):
    return x*3.2808


roomX = f2m(52/3)
roomY = f2m(49/2)

room_dim = np.r_[roomX, roomY]

m = pra.make_materials(
    ceiling="hard_surface",
    floor="hard_surface",
    east="brickwork",
    west="brickwork",
    north="brickwork",
    south="brickwork",
)

source_location = room_dim / 2 + distance * \
    np.r_[np.cos(azimuth), np.sin(azimuth)]
print(f'Source location: {source_location}')
aroom = pra.ShoeBox(room_dim, fs=fs, max_order=17,
                    materials=m, sigma2_awgn=sigma2)  # sigma2_awgn is the variance of white gaussian noise

# add the source
source_location = room_dim / 2 + distance * \
    np.r_[np.cos(azimuth), np.sin(azimuth)]
# source_signal = np.random.randn((nfft // 2 + 1) * nfft)
aroom.add_source(source_location, signal=audio)

# We use a circular array with radius 15 cm # and 12 microphones
print(f'Room center {room_dim/2}')
R = pra.circular_2D_array(room_dim / 2, 2, 0.0, 0.15)
print(f'R:\n{R}')
aroom.add_microphone_array(pra.MicrophoneArray(R, fs=aroom.fs))

# run the simulation
aroom.simulate()
# print(aroom.mic_array.signals.shape)

real_signals = np.zeros((2, len(audio)//2))
print(f'Real signals: {real_signals.shape}')
real_signals[0], real_signals[1] = split_audio(audio)
# for signal in aroom.mic_array.signals:
#     print(signal.shape)
#     print('______')
################################
# Compute the STFT frames needed
X = np.array(
    [
        pra.transform.stft.analysis(signal, nfft, nfft // 2).T
        for signal in real_signals
        # for signal in aroom.mic_array.signals
    ]
)

##############################################
# Now we can test all the algorithms available
algo_names = sorted(pra.doa.algorithms.keys())

for algo_name in algo_names:

    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    if algo_name == 'CSSM':
        doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)

        # this call here perform localization on the frames in X
        doa.locate_sources(X)

        doa.polar_plt_dirac()
        plt.title(algo_name)

        # doa.azimuth_recon contains the reconstructed location of the source
        print(algo_name)
        print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
        print("  Error:", circ_dist(azimuth, doa.azimuth_recon) /
              np.pi * 180.0, "degrees")

plt.show()

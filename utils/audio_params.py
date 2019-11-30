# Architectural constants.

NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.64  # Each example contains 50 10ms frames
#EXAMPLE_HOP_SECONDS = 0.04   # with zero overlap.
EXAMPLE_HOP_SECONDS = 0.02  # is defined dynamically to 1/(video fps)


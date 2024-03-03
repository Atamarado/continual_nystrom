import os

AUDIO_SIZE = (96, 64, 1)
GTZAN_DIR = 'data/gtzan'
GTZAN_CSV_PATH = os.path.join(GTZAN_DIR, 'features_30_sec.csv')
GTZAN_WAVEFORM_DIR = os.path.join(GTZAN_DIR, 'genres_original')
GTZAN_SAMPLING_RATE = 22050
GTZAN_LENGTH = 30  # seconds
GTZAN_SUB_LENGTH = 1  # seconds
GTZAN_SUB_HOP = 0.25  # seconds
GTZAN_VGGISH_VAL_RATIO = 0.1
GTZAN_VGGISH_TEST_RATIO = 0.1
GTZAN_VIT_VAL_RATIO = 0.18
GTZAN_VIT_TEST_RATIO = 0.1
GTZAN_SPECTROGRAMS_CACHE_PATH = 'gtzan_spectrograms.pkl'
GTZAN_FEATURES_CACHE_PATH = 'gtzan_features.pkl'
FINE_TUNED_VGGISH_PATH = 'fine_tuned_vggish.h5'
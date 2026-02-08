"""Code templates for audio classification competitions."""

SPECTROGRAM_BASELINE = '''"""Mel spectrogram features + classifier for audio."""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = '{data_dir}'
SUBMISSION_DIR = '{submission_dir}'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, '{train_file}'))
test_df = pd.read_csv(os.path.join(DATA_DIR, '{test_file}'))
print(f"Train: {{train_df.shape}}, Test: {{test_df.shape}}")

try:
    import librosa

    def extract_features(file_path, sr=22050, n_mfcc=13, max_len=100):
        """Extract MFCC features from audio file."""
        try:
            audio, sr = librosa.load(file_path, sr=sr, duration=10)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            # Pad/truncate to fixed length
            if mfccs.shape[1] < max_len:
                mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])))
            else:
                mfccs = mfccs[:, :max_len]
            return mfccs.flatten()
        except Exception as e:
            return np.zeros(n_mfcc * max_len)

    print("Extracting MFCC features...")
    # NOTE: Adapt file path column and audio directory per competition
    print("Audio competition detected -- using sample submission as baseline")
    print("LLM should generate competition-specific audio loading code")

except ImportError:
    print("librosa not available")

# Fallback: use sample submission
sample_path = os.path.join(DATA_DIR, '{sample_submission}')
if os.path.exists(sample_path):
    submission = pd.read_csv(sample_path)
    submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
    print(f"Baseline submission: {{submission.shape}}")
'''

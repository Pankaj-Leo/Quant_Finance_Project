"""
train_timegan.py
----------------
Trains a small TimeGAN on LOB microstructure data
(mid_price, spread, ofi) to generate synthetic sequences.
Compatible with ydata-synthetic >= 1.3.0
"""

import os
import warnings
import numpy as np
import tensorflow as tf


# ✅ correct import path for v1.3+
# from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN # type: ignore

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters # type: ignore

# --- Optional: Silence TensorRT and TF warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO & WARNING messages
warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)
print("ydata-synthetic version: 1.4.0")

# --- GPU check ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    print("⚠️ No GPU detected — training will run on CPU.")

# --- Load the preprocessed LOB sequence data ---
data_path = "data/processed/lob_sequences.npy"
X = np.load(data_path)
print(f"Loaded data from {data_path}")
print(f"Training data shape: {X.shape}")  # (num_sequences, seq_len, num_features)

# --- Model Parameters ---
seq_len = X.shape[1]
dim = X.shape[2]

model_parameters = ModelParameters(
    batch_size=128,
    lr=1e-4,
    noise_dim=128,
    layers_dim=256,
    latent_dim=128
)

train_parameters = TrainParameters(
    epochs=500,
    sequence_length=seq_len,  # replaces seq_len param
    number_sequences=dim       # replaces n_seq param
)

# --- Initialize and train TimeGAN ---
hidden_dim = model_parameters.layers_dim  # or any hidden size you want
gamma = 1.5  # typical value

# synthesizer = TimeGAN(
#     model_parameters=model_parameters,
#     hidden_dim=hidden_dim,
#     seq_len=seq_len,
#     n_seq=dim,
#     gamma=gamma
# )
synthesizer = TimeGAN(model_parameters)
synthesizer.seq_len = seq_len
synthesizer.n_seq = dim
print("\n🚀 Starting TimeGAN training ...\n")
# synthesizer.train(X)
train_steps = 1000  
synthesizer.train(X, train_steps)

# --- Generate synthetic sequences ---
print("\n🧪 Generating synthetic samples ...")
synthetic_data = synthesizer.sample(200)
synthetic_data = np.array(synthetic_data)
print("✅ Synthetic data shape:", synthetic_data.shape)

# --- Save outputs ---
save_path = "data/processed/lob_synthetic_timegan.npy"
np.save(save_path, synthetic_data)
print(f"✅ Saved synthetic data to {save_path}")
print("\n🎉 Training complete!")

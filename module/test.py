import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df = pd.read_parquet("./data/raw/lob_synthetic.parquet")
# print(df.head(1))
print(df.head(500))
print("Shape",df.shape)
print("Columns:", df.columns.tolist())


# Normalize 
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
print("Scaled shape:", scaled.shape)
print(scaled[:5]) 

seq_len = 100  # 100 timesteps per sequence
def make_sequences(data, seq_len):
    seqs = []
    for i in range(len(data) - seq_len):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

X = make_sequences(scaled, seq_len)
print("Final shape:", X.shape)
print(X[:1])



# np.save("data/processed/lob_sequences.npy", X)
# print("✅ Saved sequences to data/processed/lob_sequences.npy")
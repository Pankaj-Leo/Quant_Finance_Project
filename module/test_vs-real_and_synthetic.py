import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from statsmodels.tsa.stattools import acf
import pandas as pd
import seaborn as sns

# === Load data ===
real_data = np.load("data/processed/lob_sequences.npy")
synthetic_data = np.load("data/processed/lob_synthetic_timegan.npy")

print("Real:", real_data.shape)
print("Synthetic:", synthetic_data.shape)

# === 1. Clustering Check ===
pca = PCA(n_components=2)
real_pca = pca.fit_transform(real_data.reshape(real_data.shape[0], -1))
synth_pca = pca.transform(synthetic_data.reshape(synthetic_data.shape[0], -1))

kmeans_real = KMeans(n_clusters=4, random_state=42).fit(real_pca)
kmeans_synth = KMeans(n_clusters=4, random_state=42).fit(synth_pca)

ari = adjusted_rand_score(kmeans_real.labels_, kmeans_synth.labels_)
print(f"\n🌀 Adjusted Rand Index (clustering similarity): {ari:.3f}")

# plt.figure(figsize=(10,4))
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.scatter(real_pca[:,0], real_pca[:,1], c=kmeans_real.labels_, cmap='tab10', s=10)
plt.title("Real Data Clusters")

plt.subplot(1,2,2)
plt.scatter(synth_pca[:,0], synth_pca[:,1], c=kmeans_synth.labels_, cmap='tab10', s=10)
plt.title("Synthetic Data Clusters")
plt.show()

# === 2. Volatility Check ===
def rolling_volatility(series, window=10):
    return pd.Series(series).rolling(window).std().dropna().values

real_vols = [rolling_volatility(seq[:,0]) for seq in real_data[:200]]
synth_vols = [rolling_volatility(seq[:,0]) for seq in synthetic_data[:200]]

sns.kdeplot([v.mean() for v in real_vols], label='Real', fill=True)
sns.kdeplot([v.mean() for v in synth_vols], label='Synthetic', fill=True)
plt.title("Average Rolling Volatility Distribution")
plt.legend()
plt.show()

# === 3. Autocorrelation Check ===
def mean_autocorr(data, lags=20):
    acfs = []
    for seq in data[:200]:
        acfs.append(acf(seq[:,0], nlags=lags, fft=True))
    return np.mean(acfs, axis=0)

real_acf = mean_autocorr(real_data)
synth_acf = mean_autocorr(synthetic_data)

plt.plot(real_acf, label='Real')
plt.plot(synth_acf, label='Synthetic')
plt.title("Mean Autocorrelation Comparison")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()

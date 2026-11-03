import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import umap
import hdbscan

# --- 1. Setup & Data Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for Autoencoder: {str(device).upper()}")

DATA_DIR = "data"
df = pd.read_csv(os.path.join(DATA_DIR, "corpus.csv")).dropna(subset=['Abstract'])
true_labels = df['domain_label'].values
embeddings = np.load(os.path.join(DATA_DIR, "scibert_embeddings.npy"))

print(f"Loaded {embeddings.shape[0]} embeddings.")

# Neural Networks and PCA require scaled data (mean=0, variance=1)
print("Scaling embeddings...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# --- 2. The PyTorch Autoencoder Architecture ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def train_autoencoder(data, latent_dim, epochs=20, batch_size=256):
    print(f"Training Autoencoder (Latent Dim: {latent_dim}) on GPU...")
    model = Autoencoder(input_dim=768, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    tensor_data = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            latent, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        latent_representations, _ = model(tensor_data.to(device))
    return latent_representations.cpu().numpy()

# --- 3. Dimensionality Reduction (3 Methods) ---
print("\n=== Starting Dimensionality Reduction ===")

print("Running PCA...")
pca_50 = PCA(n_components=50, random_state=42).fit_transform(embeddings_scaled)
pca_2 = PCA(n_components=2, random_state=42).fit_transform(embeddings_scaled)

print("Running UMAP (This takes a few minutes)...")
umap_50 = umap.UMAP(n_components=50, random_state=42).fit_transform(embeddings_scaled)
umap_2 = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings_scaled)

ae_50 = train_autoencoder(embeddings_scaled, latent_dim=50)
ae_2 = train_autoencoder(embeddings_scaled, latent_dim=2)

# --- 4. The Elbow Method ---
print("\n=== Running Elbow Method on UMAP 50D ===")
wcss = []
k_range = range(2, 12) 
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans_test.fit(umap_50)
    wcss.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal k (UMAP 50D)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, "elbow_plot.png"))
plt.close()
print("✅ Elbow plot saved as 'elbow_plot.png'. (Assuming k=6 for the run below)")

# --- 5. Clustering & Evaluation Suite ---
print("\n=== Starting Clustering Analysis ===")
OPTIMAL_K = 6 

def evaluate_metrics(pred_labels, data_for_silhouette):
    mask = pred_labels != -1  
    if np.sum(mask) > 1:
        sil_score = silhouette_score(data_for_silhouette[mask], pred_labels[mask])
    else:
        sil_score = -1
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return round(sil_score, 4), round(ari, 4), round(nmi, 4)

results = []
reduction_methods = {
    "PCA": (pca_50, pca_2),
    "UMAP": (umap_50, umap_2),
    "Autoencoder": (ae_50, ae_2)
}

for method_name, (data_50d, data_2d) in reduction_methods.items():
    print(f"\nEvaluating {method_name} outputs...")
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
    km_labels = kmeans.fit_predict(data_50d)
    km_sil, km_ari, km_nmi = evaluate_metrics(km_labels, data_50d)
    results.append({"Reduction": method_name, "Algorithm": "K-Means", "Silhouette": km_sil, "ARI": km_ari, "NMI": km_nmi})
    
    # 2. Hierarchical (Agglomerative)
    hierarchical = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
    hier_labels = hierarchical.fit_predict(data_50d)
    hier_sil, hier_ari, hier_nmi = evaluate_metrics(hier_labels, data_50d)
    results.append({"Reduction": method_name, "Algorithm": "Hierarchical", "Silhouette": hier_sil, "ARI": hier_ari, "NMI": hier_nmi})
    
    # 3. HDBSCAN
    hdb = hdbscan.HDBSCAN(min_cluster_size=100, metric='euclidean', cluster_selection_method='eom')
    hdb_labels = hdb.fit_predict(data_2d)
    hdb_sil, hdb_ari, hdb_nmi = evaluate_metrics(hdb_labels, data_2d)
    noise_pct = round((np.sum(hdb_labels == -1) / len(hdb_labels)) * 100, 2)
    results.append({"Reduction": method_name, "Algorithm": f"HDBSCAN ({noise_pct}% Noise)", "Silhouette": hdb_sil, "ARI": hdb_ari, "NMI": hdb_nmi})

# --- Print Final Table ---
results_df = pd.DataFrame(results)
print("\n=== FINAL COMPARATIVE METRICS ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(DATA_DIR, "pipeline_comparison_metrics.csv"), index=False)
print("\n✅ All processing complete!")
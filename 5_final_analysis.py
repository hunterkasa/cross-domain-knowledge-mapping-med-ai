import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

# --- 1. Load Data ---
DATA_DIR = "data"
print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "corpus.csv")).dropna(subset=['Abstract']).reset_index(drop=True)
embeddings = np.load(os.path.join(DATA_DIR, "scibert_embeddings.npy"))

# --- 2. Run the Winning Algorithm ---
print("Running UMAP dimensionality reduction...")
# 50D for the algorithm to calculate math, 2D for us to look at the plot
umap_50d = umap.UMAP(n_components=50, random_state=42).fit_transform(embeddings)
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)

print("Running Hierarchical Clustering (k=4)...")
# Using k=4 based on your Elbow Plot
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Cluster'] = hierarchical.fit_predict(umap_50d)

# Add 2D coordinates for plotting
df['UMAP_X'] = umap_2d[:, 0]
df['UMAP_Y'] = umap_2d[:, 1]
df['Domain_Name'] = df['domain_label'].map({1: 'CS/AI', 0: 'Medicine'})

# --- 3. Cross-Domain Analysis (The Overlap) ---
print("Analyzing domain overlap per cluster...")
cluster_summary = []

for c in range(4): # For k=4
    cluster_data = df[df['Cluster'] == c]
    total_papers = len(cluster_data)
    ai_count = len(cluster_data[cluster_data['domain_label'] == 1])
    med_count = len(cluster_data[cluster_data['domain_label'] == 0])
    
    ai_pct = (ai_count / total_papers) * 100
    med_pct = (med_count / total_papers) * 100
    
    # Determine Dominance
    if ai_pct > 70: type_label = "AI-Dominant"
    elif med_pct > 70: type_label = "Med-Dominant"
    else: type_label = "Shared/Overlap"
        
    cluster_summary.append({
        "Cluster": c,
        "Total Papers": total_papers,
        "AI %": round(ai_pct, 1),
        "Medicine %": round(med_pct, 1),
        "Type": type_label
    })

summary_df = pd.DataFrame(cluster_summary)

# --- 4. Keyword Extraction (TF-IDF) ---
print("Extracting keywords...")
top_keywords = {}

# We use standard TF-IDF on the raw abstracts to find what words define each cluster
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Abstract'])
feature_names = np.array(vectorizer.get_feature_names_out())

for c in range(4):
    cluster_idx = df.index[df['Cluster'] == c].tolist()
    # Average TF-IDF scores for the cluster
    avg_tfidf = np.asarray(tfidf_matrix[cluster_idx].mean(axis=0)).flatten()
    top_indices = avg_tfidf.argsort()[-10:][::-1] # Get top 10
    top_keywords[c] = ", ".join(feature_names[top_indices])

summary_df['Top Keywords'] = summary_df['Cluster'].map(top_keywords)

print("\n=== FINAL CLUSTER SUMMARY ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(DATA_DIR, "final_cluster_topics.csv"), index=False)

# --- 5. Visualizations ---
print("Generating publication graphics...")
os.makedirs("visuals", exist_ok=True)

# Graphic 1: UMAP Scatter Plot colored by Domain
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='UMAP_X', y='UMAP_Y', 
    hue='Domain_Name', 
    palette={'CS/AI': '#ff7f0e', 'Medicine': '#1f77b4'}, 
    data=df, 
    s=5, 
    alpha=0.6, 
    edgecolor=None
)
plt.title("UMAP Projection: Computer Science vs. Medicine", fontsize=16)
plt.legend(markerscale=5)
plt.savefig("visuals/1_Domain_Map.png", dpi=300, bbox_inches='tight')
plt.close()

# Graphic 2: UMAP Scatter Plot colored by Cluster
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='UMAP_X', y='UMAP_Y', 
    hue='Cluster', 
    palette='Set1', 
    data=df, 
    s=5, 
    alpha=0.6, 
    edgecolor=None
)
plt.title("UMAP Projection: Hierarchical Clusters (k=4)", fontsize=16)
plt.legend(markerscale=5, title="Cluster")
plt.savefig("visuals/2_Cluster_Map.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Complete! Check the 'visuals' folder and the terminal output.")
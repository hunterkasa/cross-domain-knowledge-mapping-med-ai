import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
VISUALS_DIR = "visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

# --- 1. Load the Metrics ---
print("Loading comparative metrics...")
csv_path = os.path.join(DATA_DIR, "pipeline_comparison_metrics.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}. Make sure the previous script finished saving it.")
    exit()

# Clean up algorithm names (remove the specific noise percentages for cleaner labels)
df['Algorithm_Clean'] = df['Algorithm'].apply(lambda x: 'HDBSCAN' if 'HDBSCAN' in x else x)

# Create a combined label for the X-axis
df['Pipeline'] = df['Reduction'] + " + " + df['Algorithm_Clean']

# --- 2. Graphic 1: The "Winners vs Losers" Grouped Bar Chart ---
print("Generating Grouped Bar Chart...")
# Melt the dataframe so we can plot all three metrics side-by-side
df_melted = df.melt(
    id_vars=['Reduction', 'Algorithm_Clean', 'Pipeline'], 
    value_vars=['Silhouette', 'ARI', 'NMI'],
    var_name='Metric', 
    value_name='Score'
)

plt.figure(figsize=(14, 7))
sns.barplot(
    data=df_melted, 
    x='Pipeline', 
    y='Score', 
    hue='Metric',
    palette={'Silhouette': '#2ca02c', 'ARI': '#1f77b4', 'NMI': '#ff7f0e'}
)

# Formatting the chart
plt.title('Ablation Study: Clustering Performance Across All Pipelines', fontsize=16, fontweight='bold')
plt.xlabel('Pipeline (Dimensionality Reduction + Clustering)', fontsize=12)
plt.ylabel('Score (-1.0 to 1.0)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1) # Draw a line at 0
plt.legend(title='Evaluation Metric', loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "3_Ablation_BarChart.png"), dpi=300)
plt.close()


# --- 3. Graphic 2: The ARI Heatmap (Ground-Truth Accuracy) ---
print("Generating Performance Heatmap...")

# Pivot the data to create a matrix for the Heatmap (focusing on ARI)
pivot_ari = df.pivot(index='Reduction', columns='Algorithm_Clean', values='ARI')

plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot_ari, 
    annot=True,        # Show the numbers
    cmap='RdYlGn',     # Red (Loser) to Yellow to Green (Winner)
    center=0,          # Center the color scale at 0
    fmt=".3f", 
    linewidths=1, 
    linecolor='white',
    cbar_kws={'label': 'ARI Score'}
)

plt.title('Model Accuracy (ARI) by Pipeline Combination', fontsize=14, fontweight='bold')
plt.xlabel('Clustering Algorithm', fontsize=12)
plt.ylabel('Dimensionality Reduction', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "4_Ablation_Heatmap.png"), dpi=300)
plt.close()

print("✅ Complete! Check the 'visuals' folder for your ablation study charts.")
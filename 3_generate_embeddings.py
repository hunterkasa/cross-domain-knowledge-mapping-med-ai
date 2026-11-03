import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models # Added models here
import os

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# Load the corpus
DATA_DIR = "data"
corpus_path = os.path.join(DATA_DIR, "corpus.csv")
df = pd.read_csv(corpus_path)

# Remove any rows that might have empty abstracts
df = df.dropna(subset=['Abstract'])
print(f"Processing {len(df)} abstracts...")

# --- THE FIX ---
# Load the official base model and attach a mean pooling layer
word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
# ---------------

# Generate Embeddings
print("Starting embedding process (this may take a while)...")
embeddings = model.encode(
    df['Abstract'].tolist(), 
    batch_size=8,  # LOWER THIS TO PREVENT POWER SPIKES
    show_progress_bar=True, 
    convert_to_numpy=True
)

# Save the embeddings to disk
embedding_path = os.path.join(DATA_DIR, "scibert_embeddings.npy")
np.save(embedding_path, embeddings)

print(f"✅ Success! Embeddings saved to {embedding_path}")
print(f"Matrix Shape: {embeddings.shape}")
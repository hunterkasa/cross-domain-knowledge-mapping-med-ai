import os
import requests
import pandas as pd
from tqdm import tqdm
import time

# Ensure data directory exists (Windows compatible)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Using your requested naming conventions
OPENALEX_CS_FILE = os.path.join(DATA_DIR, "openalex_cs.csv")
OPENALEX_MED_FILE = os.path.join(DATA_DIR, "openalex_med.csv")

def fetch_openalex(domain_id, domain_label, source_name, output_file, target_count=30000):
    print(f"\n--- Starting OpenAlex Fetch for {source_name} ---")
    
    # We use cursor pagination for deep scraping
    base_url = "https://api.openalex.org/works"
    
    results = []
    cursor = "*"
    count = 0
    
    # Add your email to get into the "polite pool" for faster API response times
    headers = {"User-Agent": "mailto:21-44713-1@student.aiub.edu"}

    with tqdm(total=target_count, desc=f"Fetching {source_name}") as pbar:
        while count < target_count:
            params = {
                "search": "medical image segmentation", # The core sub-domain
                "filter": f"topics.domain.id:{domain_id},has_abstract:true", # Ensure it has an abstract
                "per-page": 200, # Max allowed by OpenAlex
                "cursor": cursor
            }
            
            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                items = data.get("results", [])
                if not items:
                    print("\nNo more results found.")
                    break
                    
                for item in items:
                    if count >= target_count:
                        break
                        
                    # Extract title and abstract
                    title = item.get("title", "")
                    
                    # OpenAlex returns abstracts as inverted indices, we need to reconstruct it
                    abstract_inverted = item.get("abstract_inverted_index", {})
                    abstract = ""
                    if abstract_inverted:
                        word_index = []
                        for word, positions in abstract_inverted.items():
                            for pos in positions:
                                word_index.append((pos, word))
                        # Sort by position and join into a standard string
                        word_index.sort()
                        abstract = " ".join([word for pos, word in word_index])
                    
                    if title and abstract:
                        results.append({
                            "doc_id": f"oa_{source_name.lower()}_{count}",
                            "domain_label": domain_label,
                            "Source": f"OpenAlex ({source_name})",
                            "Title": title,
                            "Abstract": abstract
                        })
                        count += 1
                        pbar.update(1)
                
                # Get the next cursor string for pagination
                cursor = data.get("meta", {}).get("next_cursor")
                if not cursor:
                    break
                    
            except Exception as e:
                print(f"\nError occurred: {e}")
                time.sleep(5) # Pause before retrying
                
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"🏁 Fetch complete. Saved {len(df)} papers to {output_file}")

if __name__ == "__main__":
    # Fetch Computer Science papers (Domain ID: domains/2) -> Label 1
    fetch_openalex(
        domain_id="domains/2", 
        domain_label=1, 
        source_name="CS", 
        output_file=OPENALEX_CS_FILE, 
        target_count=25000
    )
    
    # Fetch Medicine papers (Domain ID: domains/4) -> Label 0
    fetch_openalex(
        domain_id="domains/4", 
        domain_label=0, 
        source_name="Medicine", 
        output_file=OPENALEX_MED_FILE, 
        target_count=25000
    )
    
    print("\n--- Merging Datasets ---")
    df_cs = pd.read_csv(OPENALEX_CS_FILE)
    df_med = pd.read_csv(OPENALEX_MED_FILE)
    
    df_final = pd.concat([df_cs, df_med], ignore_index=True)
    df_final.dropna(subset=['Abstract'], inplace=True)
    
    final_path = os.path.join(DATA_DIR, "corpus.csv")
    df_final.to_csv(final_path, index=False)
    print(f"✅ Total merged dataset saved to {final_path} with {len(df_final)} abstracts.")
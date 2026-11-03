import os
from Bio import Entrez
import pandas as pd
from tqdm import tqdm
import time
from urllib.error import HTTPError

# NCBI Setup
Entrez.email = "21-44713-1@student.aiub.edu" 
DATA_DIR = "data"
PUBMED_MED_FILE = os.path.join(DATA_DIR, "pubmed_medicine_extra.csv")

def fetch_pubmed_medicine(target_count=15000):
    print(f"\n--- Fetching Extra Medicine Papers from PubMed ---")
    
    # We focus on the CLINICAL side to ensure they are "Medicine" domain
    # Excluding "Architecture" or "Framework" to avoid AI-heavy papers
    # Expanded query to catch specific modalities and clinical synonyms
    query = '("MRI"[Title/Abstract] OR "CT"[Title/Abstract] OR "ultrasound"[Title/Abstract] OR "PET"[Title/Abstract] OR "tomography"[Title/Abstract] OR "radiology"[Title/Abstract] OR "mammography"[Title/Abstract]) AND ("segmentation"[Title/Abstract] OR "contouring"[Title/Abstract] OR "delineation"[Title/Abstract] OR "annotation"[Title/Abstract])'
    
    handle = Entrez.esearch(db="pubmed", term=query, usehistory="y")
    record = Entrez.read(handle)
    webenv = record["WebEnv"]
    query_key = record["QueryKey"]
    
    # 1. Get the actual number of results found by NCBI
    total_found = int(record["Count"])
    
    # 2. Fetch either the target_count or the total_found, whichever is smaller
    fetch_count = min(total_found, target_count)
    
    print(f"NCBI found {total_found} papers matching the query.")
    print(f"Fetching {fetch_count} papers...")
    
    results = []
    batch_size = 500
    
    # 3. Safely loop only up to the available fetch_count
    for start in tqdm(range(0, fetch_count, batch_size), desc="PubMed Med"):
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed", 
                retmode="xml", 
                retstart=start, 
                retmax=batch_size, 
                webenv=webenv, 
                query_key=query_key
            )
            records = Entrez.read(fetch_handle)
            
            for article in records.get('PubmedArticle', []):
                try:
                    title = article['MedlineCitation']['Article']['ArticleTitle']
                    abstract_text = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
                    if abstract_text:
                        results.append({
                            "doc_id": f"pm_med_{len(results)}",
                            "domain_label": 0, # Medicine
                            "Source": "PubMed",
                            "Title": title,
                            "Abstract": " ".join(abstract_text)
                        })
                except Exception: 
                    continue
                    
        except HTTPError as e:
            print(f"\nHTTP Error at batch {start}: {e}. Skipping batch...")
        except Exception as e:
            print(f"\nUnexpected error at batch {start}: {e}. Skipping batch...")
            
        # Be nice to NCBI servers
        time.sleep(1)
        
    pd.DataFrame(results).to_csv(PUBMED_MED_FILE, index=False)

if __name__ == "__main__":
    # Make sure your data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    fetch_pubmed_medicine(target_count=15000)
    
    # Combine everything
    print("\n--- Finalizing Balanced Corpus ---")
    try:
        df_cs = pd.read_csv(os.path.join(DATA_DIR, "openalex_cs.csv"))
        df_med_oa = pd.read_csv(os.path.join(DATA_DIR, "openalex_med.csv"))
        df_med_pm = pd.read_csv(PUBMED_MED_FILE)
        
        df_final = pd.concat([df_cs, df_med_oa, df_med_pm], ignore_index=True)
        df_final.to_csv(os.path.join(DATA_DIR, "corpus.csv"), index=False)
        print(f"✅ Balanced Corpus Ready: {len(df_final)} total papers.")
    except FileNotFoundError as e:
        print(f"⚠️ Could not finalize corpus: {e}")
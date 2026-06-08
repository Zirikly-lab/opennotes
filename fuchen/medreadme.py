import os
import sys
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import json
from tqdm import tqdm

def calculate_medreadme_readability(input_file, text_col, output_file):
    """
    Calculate MedReadMe readability metrics and add to existing file.
    """
    
    MODEL_ID = "chaojiang06/medreadme_medical_sentence_readability_prediction_CWI"
    MAX_LEN = 512
    BATCH_SIZE = 32
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    try:
        print("Loading MedReadMe model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        ).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error during model loading: {e}")
        return None
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Filter out rows where the text column is null
    df_valid = df.dropna(subset=[text_col])
    print(f"Processing {len(df_valid)} records with valid text")
    
    # Load spaCy sentence splitter
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    def sentences_spacy_simple(text, min_chars=0, min_words=0, count_alpha_words=True):
        """
        Split text into sentences and return only those meeting char/word thresholds.
        If none meet thresholds, return original sentences (non-empty).
        """
        if not text:
            return []
        doc = nlp(text)
        original = []
        complete = []
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            original.append(s)
            if count_alpha_words:
                word_count = sum(1 for t in sent if t.is_alpha)
            else:
                word_count = sum(1 for t in sent if not t.is_space and not t.is_punct)
            char_count = len(s)
            if char_count >= min_chars and word_count >= min_words:
                complete.append(s)
        return complete if complete else original

    def score_sentences_batch(sentences, batch_size=BATCH_SIZE):
        """
        Score a list of sentences with the model in batches.
        Returns a list of floats (raw model outputs).
        """
        all_scores = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1)
            all_scores.extend([float(x) for x in logits.detach().cpu().tolist()])
        return all_scores

    def evaluate_text(text):
        """
        Evaluate a single text and return readability metrics.
        """
        if pd.isna(text) or str(text).strip() == '':
            return {
                "medreadme_score_mean": None,
                "medreadme_score_median": None
            }
        
        sentences = sentences_spacy_simple(str(text), min_chars=0, min_words=0)
        if len(sentences) == 0:
            return {
                "medreadme_score_mean": None,
                "medreadme_score_median": None
            }
        
        scores = score_sentences_batch(sentences, batch_size=BATCH_SIZE)
        scores_np = np.array(scores, dtype=float)
        
        return {
            "medreadme_score_mean": float(scores_np.mean()),
            "medreadme_score_median": float(np.median(scores_np))
        }

    # Initialize new columns with None for all rows
    medreadme_cols = ["medreadme_score_mean", "medreadme_score_median"]
    
    for col in medreadme_cols:
        df[col] = None
    
    # Process valid records
    print("Calculating MedReadMe readability metrics...")
    results = []
    
    for idx in tqdm(df_valid.index, desc="Processing texts"):
        try:
            text = df_valid.loc[idx, text_col]
            result = evaluate_text(text)
            results.append((idx, result))
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            result = {col: None for col in medreadme_cols}
            results.append((idx, result))
    
    # Update the main dataframe with results
    for idx, result in results:
        for col in medreadme_cols:
            df.loc[idx, col] = result[col]
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    valid_results = df.dropna(subset=["medreadme_score_mean"])
    if len(valid_results) > 0:
        print(f"\nSummary Statistics:")
        print(f"Records processed: {len(valid_results)}")
        print(f"Average mean score: {valid_results['medreadme_score_mean'].mean():.4f}")
        print(f"Average median score: {valid_results['medreadme_score_median'].mean():.4f}")
        print(f"Score range (mean): {valid_results['medreadme_score_mean'].min():.4f} - {valid_results['medreadme_score_mean'].max():.4f}")
        print(f"Note: Higher scores indicate less readable (more difficult) text")
    
    return df

# Run the analysis
if __name__ == "__main__":
    input_file = "../data/brief_hospital_course_with_metrics.csv"
    text_column = "brief_hospital_course"
    output_file = "../data/brief_hospital_course_with_metrics.csv" 
    
    results_df = calculate_medreadme_readability(input_file, text_column, output_file)
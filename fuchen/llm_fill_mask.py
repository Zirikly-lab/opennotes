import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM
import spacy
from tqdm import tqdm

def calculate_clinicalbert_readability(file_path, text_col, output_file):
    """
    Calculate ClinicalBERT-based readability metrics and add to existing file.
    """
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    
    # Filter out rows where the text column is null
    df_valid = df.dropna(subset=[text_col])
    print(f"Processing {len(df_valid)} records with valid text")
    
    # Load ClinicalBERT and tokenizer
    print("Loading ClinicalBERT model...")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Load spaCy for sentence splitting
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # Parameters
    mask_prob = 0.15
    max_tokens = 128  # Truncate per sentence

    # Function: masked LM on one sentence
    def evaluate_sentence(sentence):
        encoding = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_tokens)
        input_ids = encoding["input_ids"][0]
        tokens = input_ids.tolist()

        candidate_idxs = [i for i, t in enumerate(tokens) if t not in tokenizer.all_special_ids]
        if len(candidate_idxs) == 0:
            return []  # nothing to mask

        num_to_mask = max(1, int(mask_prob * len(candidate_idxs)))
        if num_to_mask > len(candidate_idxs):
            return []  # can't sample more than available

        masked_idxs = random.sample(candidate_idxs, num_to_mask)

        original_tokens = [tokens[i] for i in masked_idxs]
        masked_tokens = tokens.copy()
        for idx in masked_idxs:
            masked_tokens[idx] = tokenizer.mask_token_id

        input_tensor = torch.tensor([masked_tokens]).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0]

        probs = torch.softmax(logits, dim=-1)

        results = []
        for i, orig_token_id in zip(masked_idxs, original_tokens):
            prob_dist = probs[i]
            orig_prob = prob_dist[orig_token_id].item()
            sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True)
            rank = (sorted_indices == orig_token_id).nonzero(as_tuple=True)[0].item() + 1  # 1-based
            results.append((orig_prob, rank))
        return results

    # Function: apply to each text
    def evaluate_text_sentencewise(text):
        if pd.isna(text) or str(text).strip() == '':
            return pd.Series({"clinicalbert_avg_prob": None, "clinicalbert_avg_rank": None})
        
        doc = nlp(str(text))
        all_results = []

        for sent in doc.sents:
            results = evaluate_sentence(sent.text)
            all_results.extend(results)

        if not all_results:
            return pd.Series({"clinicalbert_avg_prob": None, "clinicalbert_avg_rank": None})
        
        avg_prob = sum(p for p, r in all_results) / len(all_results)
        avg_rank = sum(r for p, r in all_results) / len(all_results)
        return pd.Series({"clinicalbert_avg_prob": avg_prob, "clinicalbert_avg_rank": avg_rank})

    # Apply to all valid records
    print("Calculating ClinicalBERT readability metrics...")
    tqdm.pandas()
    
    # Initialize columns with None for all rows
    df["clinicalbert_avg_prob"] = None
    df["clinicalbert_avg_rank"] = None
    
    # Calculate metrics only for valid text rows
    valid_indices = df_valid.index
    results = df_valid[text_col].progress_apply(evaluate_text_sentencewise)
    
    # Update the main dataframe with results
    df.loc[valid_indices, "clinicalbert_avg_prob"] = results["clinicalbert_avg_prob"]
    df.loc[valid_indices, "clinicalbert_avg_rank"] = results["clinicalbert_avg_rank"]

    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    valid_results = df.dropna(subset=["clinicalbert_avg_prob"])
    if len(valid_results) > 0:
        print(f"\nSummary Statistics:")
        print(f"Records processed: {len(valid_results)}")
        print(f"Average probability: {valid_results['clinicalbert_avg_prob'].mean():.4f}")
        print(f"Average rank: {valid_results['clinicalbert_avg_rank'].mean():.2f}")
        print(f"Probability range: {valid_results['clinicalbert_avg_prob'].min():.4f} - {valid_results['clinicalbert_avg_prob'].max():.4f}")
        print(f"Rank range: {valid_results['clinicalbert_avg_rank'].min():.0f} - {valid_results['clinicalbert_avg_rank'].max():.0f}")
    
    return df

# Run the analysis
if __name__ == "__main__":
    input_file = "../data/brief_hospital_course_with_metrics.csv"
    text_column = "brief_hospital_course"
    output_file = "../data/brief_hospital_course_with_metrics.csv" 
    
    results_df = calculate_clinicalbert_readability(input_file, text_column, output_file)
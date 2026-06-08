import pandas as pd
import os
from wordfreq import word_frequency, zipf_frequency
import spacy
from tqdm import tqdm

# === Config ===
# Load spaCy model (ensure you have run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  SpaCy model not found. Downloading 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

OUTPUT_DIR = "../data/results_with_metrics"

# File configurations: (Input Path, Column Name to Analyze)
FILES_TO_PROCESS = [
    ("/projects/opennotes/fli49/OpenNotes/data/results_with_metrics/processed_brief_hospital_course_with_metrics.csv", "brief_hospital_course"),
    ("/projects/opennotes/fli49/OpenNotes/data/results_with_metrics/processed_discharge_instructions_with_metrics.csv", "discharge_instructions")
]

# === Define Metric Functions ===

def get_wordfreq_metrics(text):
    """
    Calculates three versions of word frequency metrics for a given text.
    Returns a dictionary of results.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'mean_freq_original': 0.0,
            'mean_freq_improved': 0.0,
            'mean_zipf_freq': 0.0
        }

    # Process text with spaCy (disable NER/parser for speed)
    doc = nlp(text.lower())
    
    # 1. Original Method List (raw text, is_alpha)
    words_original = [token.text for token in doc if token.is_alpha]
    
    # 2. Improved/Zipf Method List (lemma, is_alpha)
    words_lemma = [token.lemma_ for token in doc if token.is_alpha]

    # --- Calculation 1: Original ---
    if words_original:
        freqs_orig = [word_frequency(w, 'en') for w in words_original]
        score_orig = sum(freqs_orig) / len(freqs_orig)
    else:
        score_orig = 0.0

    # --- Calculation 2: Improved (Large wordlist, ignore zeros) ---
    if words_lemma:
        freqs_imp = [word_frequency(w, 'en', wordlist='large') for w in words_lemma]
        nonzero_freqs = [f for f in freqs_imp if f > 0]
        # Note: Your original code divided sum(nonzero) by len(words), preserving the penalty for OOVs
        score_imp = sum(nonzero_freqs) / len(words_lemma)
    else:
        score_imp = 0.0

    # --- Calculation 3: Zipf (Logarithmic) ---
    if words_lemma:
        zipfs = [zipf_frequency(w, 'en', wordlist='large') for w in words_lemma]
        score_zipf = sum(zipfs) / len(words_lemma)
    else:
        score_zipf = 0.0

    return {
        'mean_freq_original': score_orig,
        'mean_freq_improved': score_imp,
        'mean_zipf_freq': score_zipf
    }

# === Processing Loop ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_path, target_col in FILES_TO_PROCESS:
    filename = os.path.basename(file_path)
    print(f"\n🚀 Processing lexical complexity for: {filename}")
    
    # Load Data
    if os.path.exists(os.path.join(OUTPUT_DIR, f"processed_{filename}")):
        # If we already processed this file in the previous step (QuickUMLS), load that version
        print(f"   ℹ️  Loading previously processed file from {OUTPUT_DIR}...")
        load_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")
    else:
        # Fallback to original if the processed one doesn't exist
        print(f"   ℹ️  Loading original file...")
        load_path = file_path

    try:
        df = pd.read_csv(load_path)
    except FileNotFoundError:
        print(f"❌ File not found: {load_path}")
        continue

    # Run Extraction
    tqdm.pandas(desc=f"   Calculating frequencies")
    metrics_df = df[target_col].progress_apply(get_wordfreq_metrics).apply(pd.Series)

    # Concatenate metrics back to DF
    # We use a suffix in case these columns already exist to avoid errors
    result_df = pd.concat([df, metrics_df], axis=1)

    # Overwrite the processed file (or create new)
    output_file = os.path.join(OUTPUT_DIR, f"processed_{filename}")
    result_df.to_csv(output_file, index=False)
    
    print(f"✅ Updated file saved to: {output_file}")

print("\n🎉 Complexity analysis complete.")
import pandas as pd
import textstat
import spacy
import numpy as np
from collections import Counter
from tqdm import tqdm

def calculate_text_metrics(file_path, text_col = 'text'):
    """
    Calculate readability and linguistic metrics for medical notes.
    """
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} notes")
    
    # Load spaCy English model
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Calculate readability metrics
    print("Calculating readability metrics...")
    df["SMOG"] = df[text_col].apply(textstat.smog_index)
    df["KFGL"] = df[text_col].apply(textstat.flesch_kincaid_grade)
    df["GFI"] = df[text_col].apply(textstat.gunning_fog)
    
    # Initialize lists for linguistic metrics
    sentence_counts = []
    word_counts = []
    entropies = []
    
    # Process notes with spaCy
    print("Processing notes with spaCy...")
    for text in tqdm(df[text_col], desc="Processing notes"):
        # Process text with spaCy
        doc = nlp(str(text))

        # Get sentences
        sentences = list(doc.sents)
        sentence_counts.append(len(sentences))

        # Get word tokens (exclude punctuation and spaces)
        words = [token.text.lower() for token in doc if token.is_alpha]
        word_counts.append(len(words))

        # Compute Shannon entropy
        if words:
            word_freq = Counter(words)
            probs = np.array([freq / len(words) for freq in word_freq.values()])
            entropy = -np.sum(probs * np.log2(probs))
        else:
            entropy = 0.0
        entropies.append(entropy)

    # Add linguistic metrics to DataFrame
    df['num_sentences'] = sentence_counts
    df['num_words'] = word_counts
    df['shannon_entropy'] = entropies
    
    # Save the results
    output_file = file_path.replace('.csv', '_with_metrics.csv')
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    return df

# Run the analysis
if __name__ == "__main__":
    file_path = "../data/brief_hospital_course.csv"
    results_df = calculate_text_metrics(file_path, text_col = 'brief_hospital_course')
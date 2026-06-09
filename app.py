import gradio as gr
from fuchen.word_frequency import get_wordfreq_metrics
import spacy
import pandas as pd
import re
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Load spaCy model (download if needed: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Simplified CHV mapping (sample - expand as needed)
CHV_MAPPING = {
    "myocardial infarction": "heart attack",
    "hypertension": "high blood pressure",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
    "cerebrovascular accident": "stroke",
    "pneumonia": "lung infection",
    "edema": "swelling",
    "tachycardia": "fast heart rate",
    "bradycardia": "slow heart rate",
    "dyspnea": "shortness of breath"
}

# Common medical abbreviations
COMMON_ABBREVIATIONS = {
    "MRI": "magnetic resonance imaging",
    "CT": "computed tomography",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "SOB": "shortness of breath",
    "CHF": "congestive heart failure",
    "MI": "myocardial infarction",
    "COPD": "chronic obstructive pulmonary disease",
    "DM": "diabetes mellitus",
    "HTN": "hypertension",
    "CVA": "cerebrovascular accident"
}

# Sample eponyms and brand names (expand as needed)
EPONYMS = {"Parkinson", "Alzheimer", "Crohn", "Hodgkin", "Addison", "Cushing"}
BRAND_NAMES = {"Tylenol", "Advil", "Motrin", "Lipitor", "Zoloft", "Prozac"}

def calculate_ttr(text):
    """Calculate Type-Token Ratio"""
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0, 0, 0
    types = set(tokens)
    ttr = len(types) / len(tokens)
    return len(tokens), len(types), ttr

def match_chv_terms(text):
    """Match terms to Consumer Health Vocabulary"""
    text_lower = text.lower()
    technical_terms = []
    lay_terms = []
    
    for tech, lay in CHV_MAPPING.items():
        if tech in text_lower:
            technical_terms.append(tech)
            lay_terms.append(lay)
    
    if technical_terms:
        coverage = (len(set(technical_terms)) / len(CHV_MAPPING)) * 100
    else:
        coverage = 0
    
    return {
        "chv_technical_terms_found": len(set(technical_terms)),
        "chv_lay_alternatives": len(set(lay_terms)),
        "chv_coverage_percent": round(coverage, 2)
    }

def calculate_medical_density(text):
    """Calculate UMLS concept coverage and SNOMED density using spaCy"""
    doc = nlp(text)
    
    # Medical entities (simplified - using spaCy entity recognition)
    medical_entities = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]  # Filter medical-specific
    # Alternative: look for medical terms using word properties
    medical_terms = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]
    
    total_tokens = len([token for token in doc])
    if total_tokens == 0:
        density = 0
    else:
        density = (len(set(medical_terms)) / total_tokens) * 100
    
    return {
        "umls_concepts_found": len(set(medical_entities)),
        "medical_terms_count": len(set(medical_terms)),
        "medical_terminology_density": round(density, 2)
    }

def analyze_abbreviations(text):
    """Analyze abbreviation frequency and standardization"""
    # Find potential abbreviations (2-5 uppercase letters)
    pattern = r'\b[A-Z]{2,5}\b'
    found_abbrev = re.findall(pattern, text)
    
    # Find abbreviations with definitions (e.g., "MRI (magnetic resonance imaging)")
    defined_pattern = r'\b([A-Z]{2,5})\s*\(([^)]+)\)'
    defined = re.findall(defined_pattern, text)
    
    total_tokens = len(text.split())
    if total_tokens == 0:
        freq = 0
    else:
        freq = len(found_abbrev) / total_tokens
    
    # Standardization rate (how many match common medical abbreviations)
    standardized = sum(1 for ab in set(found_abbrev) if ab in COMMON_ABBREVIATIONS)
    if set(found_abbrev):
        std_rate = (standardized / len(set(found_abbrev))) * 100
    else:
        std_rate = 0
    
    return {
        "abbreviation_count": len(found_abbrev),
        "unique_abbreviations": len(set(found_abbrev)),
        "abbreviation_frequency": round(freq, 4),
        "standardization_rate": round(std_rate, 2),
        "defined_abbreviations": len(defined)
    }

def analyze_eponyms_brands(text):
    """Analyze eponym and brand name usage patterns"""
    words = text.split()
    
    # Find eponyms (capitalized words that might be names)
    eponym_pattern = r'\b([A-Z][a-z]+(?:\'s)?)\b'
    found_eponyms = [m for m in re.findall(eponym_pattern, text) if m in EPONYMS]
    
    # Find brand names
    found_brands = [word for word in words if word in BRAND_NAMES]
    
    total_tokens = len(words)
    if total_tokens == 0:
        eponym_rate = brand_rate = 0
    else:
        eponym_rate = (len(found_eponyms) / total_tokens) * 1000  # per 1000 tokens
        brand_rate = (len(found_brands) / total_tokens) * 1000
    
    return {
        "eponym_count": len(found_eponyms),
        "unique_eponyms": len(set(found_eponyms)),
        "eponyms_per_1000_tokens": round(eponym_rate, 2),
        "brand_name_count": len(found_brands),
        "brand_names_per_1000_tokens": round(brand_rate, 2)
    }

def annotate(text):
    """Main annotation function with all metrics"""
    if not text or text.strip() == "":
        return [[
            "note_001", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]]
    
    note_id = "note_001"
    word_count = len(text.split())
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    
    # Word frequency metrics
    wordfreq_metrics = get_wordfreq_metrics(text)
    
    # 1. Vocabulary metrics
    tokens, types, ttr = calculate_ttr(text)
    
    # 2. CHV matching
    chv_metrics = match_chv_terms(text)
    
    # 3. Medical density
    medical_metrics = calculate_medical_density(text)
    
    # 4. Abbreviation analysis
    abbrev_metrics = analyze_abbreviations(text)
    
    # 5. Eponym and brand analysis
    eponym_metrics = analyze_eponyms_brands(text)
    
    # Return all metrics in a flat list
    return [[
        note_id,
        word_count,
        sentence_count,
        wordfreq_metrics['mean_freq_original'],
        wordfreq_metrics['mean_freq_improved'],
        wordfreq_metrics['mean_zipf_freq'],
        tokens,
        types,
        round(ttr, 4),
        chv_metrics['chv_technical_terms_found'],
        chv_metrics['chv_lay_alternatives'],
        chv_metrics['chv_coverage_percent'],
        medical_metrics['medical_terminology_density'],
        medical_metrics['umls_concepts_found'],
        abbrev_metrics['abbreviation_frequency'],
        abbrev_metrics['standardization_rate'],
        eponym_metrics['eponyms_per_1000_tokens'],
        eponym_metrics['brand_names_per_1000_tokens'],
        eponym_metrics['eponym_count'] + eponym_metrics['brand_name_count']
    ]]

# Create Gradio interface
with gr.Blocks(title="EHR Annotation Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # EHR Annotation Tool
    
    Provide a medical note to annotate:
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                placeholder="Enter a medical note here...\n\nExample: Patient presents with hypertension and SOB. MRI scheduled for tomorrow. No signs of myocardial infarction.",
                lines=15,
                label="Medical Progress Note"
            )
            submit_btn = gr.Button("Annotate Note", variant="primary", size="lg")
        
        # with gr.Column(scale=1):
        #     gr.Markdown("### 📊 Quick Stats")
        #     example_btn = gr.Button("📋 Load Example", size="sm")
    
    output_table = gr.Dataframe(
        headers=[
            "Note ID", "Word Count", "Sentence Count",
            "Mean Freq (Original)", "Mean Freq (Improved)", "Mean Zipf Freq",
            "Tokens", "Types", "TTR",
            "CHV Terms Found", "CHV Lay Alts", "CHV Coverage %",
            "Medical Density %", "UMLS Concepts",
            "Abbrev Frequency", "Abbrev Std Rate %",
            "Eponyms/1K", "Brands/1K", "Total Names"
        ],
        label="Annotation Results",
        interactive=False,
        wrap=True,
        # max_cols=20
    )
    
    # example_btn.click(fn=load_example, outputs=input_text)
    submit_btn.click(fn=annotate, inputs=input_text, outputs=output_table)

if __name__ == "__main__":
    demo.launch()
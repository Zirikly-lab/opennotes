---
title: Opennotes
emoji: 🐠
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.15.2
python_version: '3.13'
app_file: app.py
pinned: false
---


A huggingface space code for an mvp version of the EHR annotation tool we are building. This readme includes a todo of the features we are implemented and their status.

Try it live: huggingface.co/spaces/saadsorganization/opennotes

You have to be part of my Saad's Organization other wise it is prive. Send an invite to join.
# Features to Implement

## Structural features: 
[X] Note/Document length 
[X] Document length distributions (tokens, sentences, sections) 

## Lexical & Semantic features: 
Match terms to the Consumer Health Vocabulary52 for the medical/technical vs. lay terms
Vocabulary size and type-token ratios 
Medical terminology density (UMLS concept coverage and SNOMED) 
Abbreviation frequency and standardization 
Eponym and brand name usage patterns (if we have time)

## Redundancy & Information Dynamics
Identify Copy-Forward portions of notes 
Remove copy-forward portion
Other types of duplicates 
Measure new information through the variation in words in a note, using Shannon entropy, compared to average variations between subsequent patient encounters.

## Information density: 
- Unique clinical concepts per 100 words - New information vs. repeated information ratios - Signal-to-noise estimates (clinical content vs. boilerplate)

# Tool input/output
Input:
Clinical note from MIMIC3 or 4
Clinical note from JHU data

Output
CSV file with all of the output fields
JSON file with all of the output fields
Other options?


| Feature | How It Was Calculated | Notes |
|---------|----------------------|-------|
| Word Count | `len(text.split())` | None |
| Sentence Count | Count of `.`, `!`, `?` | Misses abbreviations like "Dr." |
| Mean Frequency (Original/Improved/Zipf) | `wordfreq` package lookups | None |
| Tokens | `nltk.word_tokenize()` | None |
| Types | `set(tokens)` | None |
| Type-Token Ratio (TTR) | `len(types) / len(tokens)` | TTR drops with longer texts; use MATTR for production |
| CHV Terms Found | Dictionary lookup vs. predefined technical terms | Full CHV has 50K terms → brute force O(n*m) fails. Use Aho-Corasick |
| CHV Coverage % | `(matched_terms / total_chv_terms) * 100` | Only meaningful if CHV is complete |
| Medical Terminology Density | `(medical_nouns / total_tokens) * 100` using spaCy POS | Overcounts non-medical nouns ("patient") |
| UMLS Concepts Found | Placeholder - counts generic spaCy entities | **Not real UMLS**. Real = QuickUMLS (8GB RAM, 1K notes/min) or scispacy |
| Abbreviation Frequency | Regex `\b[A-Z]{2,5}\b` | Misses lowercase clinical abbrevs ("bid", "prn") |
| Abbreviation Standardization Rate | Match against 12 common abbrevs | Needs full UMLS abbrev database (50K pairs) |
| Eponyms per 1K Tokens | Match against 5 eponyms (Parkinson, etc.) | Toy list. Full detection requires NER + KB lookup |
| Brand Names per 1K Tokens | Match against 6 drug names | Toy list. Full list = RxNorm (100K+ names) needs NER |

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

| Feature | How It Was Calculated | Approach | Notes |
|---------|----------------------|----------|-------|
| Word Count | `len(text.split())` | Simple | None |
| Sentence Count | Count of `.`, `!`, `?` | Simple | Misses abbreviations like "Dr." |
| Mean Frequency (Original/Improved/Zipf) | `wordfreq` package | Library | None |
| Tokens | `nltk.word_tokenize()` | Library | None |
| Types | `set(tokens)` | Simple | None |
| Type-Token Ratio (TTR) | `len(types)/len(tokens)` | Simple | Use MATTR for long texts |
| CHV Terms Found | Dict lookup vs 10 hardcoded terms | **Needs: UMLS license + CHV subset + Aho-Corasick** | Brute force fails at 50K terms |
| Medical Terminology Density | Counts all nouns via spaCy POS | **Needs: UMLS license + QuickUMLS or scispacy** | Current method is wrong (counts "patient" as medical) |
| UMLS Concepts Found | Placeholder - generic NER | **Needs: UMLS license + MRCONSO.RRF + QuickUMLS (8GB RAM)** | Not implemented |
| Abbreviation Frequency | Regex `\b[A-Z]{2,5}\b` | Simple | Misses "bid", "prn" |
| Abbreviation Standardization Rate | Match vs 12 hardcoded | **Needs: UMLS abbreviation database** | Toy list |
| Eponyms per 1K Tokens | Match vs 5 hardcoded names | **Needs: Complete eponyms list + NER filtering** | Toy list |
| Brand Names per 1K Tokens | Match vs 6 hardcoded drugs | **Needs: RxNorm database + drug NER** | Toy list |


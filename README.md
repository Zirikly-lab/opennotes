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


# Features
## Structural features: 
Note/Document length 
Document length distributions (tokens, sentences, sections) 

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

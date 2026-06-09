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

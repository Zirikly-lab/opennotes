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


| Feature | How It Was Calculated | Notes & Scaling Considerations |
|---------|----------------------|-------------------------------|
| **Word Count** | `len(text.split())` | Trivial. Scales O(n). |
| **Sentence Count** | Count of `.`, `!`, `?` characters | Inaccurate for abbreviations (e.g., "Dr.", "e.g."). Scaling: use spaCy sentence tokenizer for production. |
| **Mean Frequency (Original)** | From `wordfreq` package - raw frequency per million | O(n) lookups. Scaling: wordfreq uses precomputed database, fine for ~10K notes. Memory: ~50MB. |
| **Mean Frequency (Improved)** | From `wordfreq` - weighted by corpus | Same as above. |
| **Mean Zipf Frequency** | From `wordfreq` - Zipf scale (1-7) | Same as above. |
| **Tokens** | `nltk.word_tokenize()` | O(n). Scaling: NLTK is slow on large docs (10M+ tokens). Replace with `spaCy` tokenizer (2-3x faster). |
| **Types** | `set(tokens)` | O(n) time, O(unique tokens) memory. Scaling: for 100K notes, types set could hit 500K+ items → memory pressure. Use streaming/bloom filters. |
| **Type-Token Ratio (TTR)** | `len(types) / len(tokens)` | Simple. But TTR is length-sensitive (shorter texts = higher TTR). Scaling: use **MATTR** (Moving Average TTR) for robust cross-doc comparison. |
| **CHV Technical Terms Found** | Dictionary lookup of lowercase text for predefined technical terms (e.g., "myocardial infarction") | O(k * n) where k = CHV size. Current CHV = ~10 terms → fine. **Scaling hazard**: Full CHV has ~50K terms. Each note would do 50K substring checks → O(50K * note_length). **Fix**: Use Aho-Corasick automaton or regex trie for O(n + m) time. Also requires full CHV data license from UMLS. |
| **CHV Lay Alternatives** | Count of unique lay terms mapped from matched technical terms | Same as above. |
| **CHV Coverage %** | `(unique_technical_matches / total_chv_terms) * 100` | Only meaningful if CHV is complete. Scaling: coverage becomes sparse (most terms not in a single note). Consider **recall@k** instead. |
| **Medical Terminology Density** | `(unique_medical_nouns_propn / total_tokens) * 100` where medical = spaCy POS tags (NOUN, PROPN) with length > 2 | Oversimplified (captures non-medical nouns like "patient", "room"). Scaling: spaCy POS tagging is O(n) but ~100K tokens/sec on CPU. Better: Use **scispacy** + UMLS lookup → 10x slower but accurate. Density formula itself is fine. |
| **UMLS Concepts Found** | Count of unique spaCy named entities (filtered by PERSON/ORG/GPE/DATE) | **This is incorrect** - those are generic NER, not UMLS concepts. Scaling: Proper UMLS extraction requires **QuickUMLS** (faster, ~1K notes/min) or **SciSpacy + UMLS** (slower, ~200 notes/min). QuickUMLS needs 8GB RAM for index. For prototype, this is a placeholder. |
| **Abbreviation Frequency** | `(abbrev_count / total_tokens)` where abbrev = `\b[A-Z]{2,5}\b` regex | O(n) regex. Misses lowercase abbrevs (e.g., "bid", "prn"). Scaling: Regex is fast ( <1ms per KB). Consider **Ab3P** or **Schwartz-Hearst** algorithm for definition detection (10x slower but accurate). |
| **Abbreviation Standardization Rate** | `(abbrevs_in_standard_list / unique_abbrevs) * 100` with 12 common medical abbrevs | Depends heavily on list completeness. Scaling: Need full UMLS abbreviation database or **PubMed abbreviations** (~50K pairs). Then O(log n) lookup per abbrev with hash map - fine. |
| **Eponyms per 1,000 Tokens** | `(eponym_count / total_tokens) * 1000` where eponym = match against set of 5 names (Parkinson, Alzheimer, etc.) | Current set is toy (5 eponyms). Scaling: Need full **eponym list** (~2,000 names from medical naming conventions). Use **fasttext** or **Word2Vec** for name detection? Eponym detection is hard - many false positives (e.g., "Thomas", "Jones" as patient names vs. "Thomas splint"). Production: Use NER to filter person names, then check if they appear in known eponym KB. |
| **Brand Names per 1,000 Tokens** | `(brand_count / total_tokens) * 1000` against set of 6 drug brand names | Same scaling issue as eponyms. Full **RxNorm** has 100K+ brand names. Detection: Use **scispacy drug NER** or **ChemNER** (slower but accurate). Brand names change over time → need monthly updates. |
| **Total Names** | Sum of eponym + brand name counts | Simple addition. No scaling issues. |


# Learning from LLM Disagreement in Retrieval Evaluation
Code and Data for Ingram et al., JCDL 2025

## Overview

This repository contains code for reproducing the experiments in:  

Ingram, W. A., Banerjee, B., and Fox, E. A. (2025). “Learning from LLM Disagreement in Retrieval Evaluation.” Proceedings of JCDL 2025.  

The repository provides scripts and notebooks for:  
	•	Building TF–IDF representations for SDG-retrieved abstracts.  
	•	Isolating agreement and disagreement subsets for LLaMA and Qwen relevance labels.  
	•	Computing lexical divergence, permutation statistics, and KL divergence.  
	•	Simulating centroid-based and representative-query retrieval over disagreement sets.  
	•	Evaluating learnability of disagreement using logistic regression and ROC curves.  
	•	Cross-referencing disagreement rows with student-ready teacher scores to test whether disagreement cases sit in low-confidence regions.  

All code is implemented in Python, using scikit-learn for vectorization and modeling.

## Additional Analysis Script

The repository now includes a standalone script for linking disagreement rows
back to the original per-SDG Qwen probability CSVs and summarizing the
corresponding teacher probability (`p1`) and teacher logit (`teacher_logit`)
scores:

```bash
uv sync
python scripts/analyze_disagreement_scores.py
```

By default, the script reads:

- `data/model_disagreements.csv`
- `data/sdg1xsdg1_2023_train__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv`
- `data/sdg1xsdg1_2023_test__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv`
- `data/sdg3xsdg3_2023_train__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv`
- `data/sdg3xsdg3_2023_test__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv`
- `data/sdg7xsdg7_2023_train__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv`
- `data/sdg7xsdg7_2023_test__scopus_sdg1_qwen_binary_bit_with_probs_v1.csv`

and writes CSV outputs under:

- `data/analysis/disagreement_scores/`

The main outputs are:

- `disagreement_score_rows.csv`: row-level disagreement cases with the matched
  per-SDG `p1` and `teacher_logit` scores from the original Qwen scoring CSVs
- `disagreement_score_summary.csv`: grouped summaries by SDG and disagreement
  direction
- `disagreement_score_summary_by_split.csv`: the same summaries broken out by
  train/test split
- `disagreement_score_baseline_by_sdg.csv`: baseline score distributions for
  the full per-SDG Qwen scoring corpus

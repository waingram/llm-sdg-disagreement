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

All code is implemented in Python, using scikit-learn for vectorization and modeling.

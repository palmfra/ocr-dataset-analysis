# OpenCodeReasoning Dataset Analysis

This repository contains a comprehensive analysis of the [OpenCodeReasoning](https://huggingface.co/datasets/THUDM/OpenCodeReasoning) dataset. The dataset contains natural language programming problems and is designed to test code understanding, reasoning, and generation capabilities.

## Project Goals

- Explore and understand the structure and diversity of the dataset.
- Develop and evaluate models for:
  - Difficulty classification
  - Code generation
  - Bug detection

---

## Dataset Summary

- **Subset Used**: 10% of the original dataset (excluding "UNKNOWN_DIFFICULTY" labels).
- **Fields Analyzed**: `input` (problem statement), `output` (expected result), `solution` (reference code).
- **Observations**:
  - Varied problem lengths and vocabulary.
  - Dominance of medium/hard difficulty problems.
  - Mixture of natural language and code requires hybrid NLP techniques.

---

## Analysis Overview

### 1. Clustering

- **Method**: TF-IDF + K-Means on problem texts.
- **Result**: 5 thematic clusters (e.g., algorithms, strings, data structures).
- **t-SNE**: Visualized meaningful cluster separation.

### 2. Information Retrieval

- **Tools**: PyTerrier with TF, TF-IDF, BM25.
- **Use Case**: Retrieve similar problems for few-shot examples.
- **Outcome**: BM25 showed best recall; hybrid ranks improved results.

### 3. Word Embedding Training

- **Model**: Word2Vec (30D, window=10).
- **Findings**: Captured semantic groupings (e.g., `"loop"` ≈ `"break"`).
- **Limitation**: Small training sample size limited coverage.

---

## Difficulty Classification

- **Objective**: Predict difficulty level (Easy, Medium, Hard, Very Hard).
- **Methods**:
  - Logistic Regression (TF-IDF): 87% accuracy
  - DistilBERT (fine-tuned): 97% accuracy
  - TinyLlama + LoRA: 54% (few-shot)

- **Insights**:
  - Transformers outperformed traditional methods.
  - Contextual clues (e.g., "O(n)", "recursion") are key indicators.
  - Label noise and class imbalance remain challenges.

---

## Code Generation

- **Goal**: Generate Python code from problem statements.

- **Approaches**:
  - **Fine-Tuning CodeGen**: Overfitted, high compute, poor performance.
  - **Zero-Shot with DeepSeek-Coder 6.7B Instruct**:
    - High similarity to reference code (cosine ≈ 0.997)
    - Good for simple/medium tasks.
  - **Few-Shot Prompting**:
    - Further improved accuracy and constraint handling
    - Cosine similarity ≈ 0.992

- **Best Approach**: Few-shot prompting with DeepSeek-Coder 6.7B Instruct.

---

## Bug Detection

- **Goal**: Detect and classify bugs in Python code.

- **Data**: Synthetic bugs injected (10 types, 50% probability).

- **Models**:
  - **Binary Classification (CodeBERT)**: 89% accuracy
  - **Multi-Label Classification (CodeBERT)**:
    - Micro F1: 0.81
    - Best on indentation/missing returns
    - Struggled with subtle errors (e.g., off-by-one)

- **Limitations**:
  - Synthetic bugs lack real-world nuance.
  - Some bug types (like operator swaps) are hard to detect.

---

## Key Findings

- **Transformer models** (e.g., DistilBERT, CodeBERT) consistently outperformed simpler models.
- **Few-shot prompting** with instruction-tuned models is effective for code generation.
- **Bug detection** benefits from multi-label modeling, but real-world data is needed for better generalization.
- **Dataset diversity** supports multi-task training and retrieval-augmented workflows.

---

## License

This project is released under the MIT License.

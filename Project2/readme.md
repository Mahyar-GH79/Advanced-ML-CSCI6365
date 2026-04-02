# 📰 News Feed Recommendation on the MIND Dataset

A comprehensive end-to-end news recommendation system built on the **Microsoft News Dataset (MIND-small)**. This project implements and compares **12 ranking models** across six families — from a simple popularity baseline to pretrained transformer encoders — evaluated with standard information retrieval metrics.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview

Given a user's reading history and a set of candidate news articles, the goal is to rank the articles the user is most likely to click at the top. We implement 12 recommendation approaches of increasing sophistication and compare them head-to-head on the MIND-small benchmark.

**Key findings:**
- 🥇 **Sentence-BERT** is the best single model (AUC = 0.6356), outperforming even the collaborative hybrid
- ⚠️ **Pretrained transformers** (BERT, RoBERTa, DistilBERT) with raw [CLS] pooling *underperform* TF-IDF — model size ≠ quality without fine-tuning
- 🤝 The **Hybrid (TF-IDF + SVD)** is the strongest practical model that requires no GPU
- 📉 **Neighborhood-based methods** (User-CF, Clustering) fall below the popularity baseline due to interaction sparsity

---

## Models Implemented

### Collaborative & Hybrid
| # | Model | Description |
|---|-------|-------------|
| 1 | **Popularity Baseline** | Global click frequency — non-personalized floor |
| 2 | **User-Based CF** | K=20 nearest neighbors via cosine similarity on binary interaction matrix |
| 3 | **SVD (Matrix Factorization)** | Truncated SVD with k=100 latent components |
| 4 | **Clustering (K-Means)** | K=50 user clusters on SVD latent vectors + per-category CTR |
| 5 | **Hybrid (TF-IDF + SVD)** | Weighted combination (α=0.6) of content and collaborative scores |

### Content-Based — Lightweight Encoders
| # | Model | Encoder | Dim |
|---|-------|---------|-----|
| 6 | **TF-IDF** | Bag-of-words, unigrams+bigrams, sublinear TF | 30,000 (sparse) |
| 7 | **Word2Vec** | Trained from scratch on article corpus, mean pooling | 100 |
| 8 | **FastText** | Subword-aware, trained from scratch, mean pooling | 100 |
| 9 | **Sentence-BERT** | `all-MiniLM-L6-v2`, contrastive fine-tuned | 384 |

### Content-Based — Pretrained Transformer Encoders
| # | Model | HuggingFace ID | Params | Dim |
|---|-------|---------------|--------|-----|
| 10 | **BERT** | `bert-base-uncased` | 110M | 768 |
| 11 | **RoBERTa** | `roberta-base` | 125M | 768 |
| 12 | **DistilBERT** | `distilbert-base-uncased` | 66M | 768 |

All content-based models share the same inference pattern: user profile = mean embedding of click history → rank candidates by cosine similarity.

---

## Results

Results on the MIND-small validation set (3,000 sampled impressions):

| Rank | Model | AUC | MRR@10 | nDCG@5 | nDCG@10 |
|------|-------|-----|--------|--------|---------|
| 🥇 1 | Content-Based (SBERT) | **0.6356** | **0.3375** | **0.3301** | **0.3921** |
| 2 | Hybrid (CB + SVD) | 0.6260 | 0.3361 | 0.3299 | 0.3884 |
| 3 | SVD (Matrix Factorization) | 0.5991 | 0.3235 | 0.3157 | 0.3736 |
| 4 | Content-Based (TF-IDF) | 0.5968 | 0.3218 | 0.3135 | 0.3717 |
| 5 | Popularity Baseline | 0.5955 | 0.3192 | 0.3100 | 0.3659 |
| 6 | Content-Based (DistilBERT) | 0.5822 | 0.2981 | 0.2951 | 0.3536 |
| 7 | Content-Based (RoBERTa) | 0.5731 | 0.2940 | 0.2849 | 0.3441 |
| 8 | Content-Based (Word2Vec) | 0.5620 | 0.2782 | 0.2738 | 0.3310 |
| 9 | Clustering (K-Means) | 0.5529 | 0.2636 | 0.2606 | 0.3208 |
| 10 | User-Based CF | 0.5500 | 0.2915 | 0.2802 | 0.3364 |
| 11 | Content-Based (BERT) | 0.5474 | 0.2772 | 0.2689 | 0.3285 |
| 12 | Content-Based (FastText) | 0.5466 | 0.2714 | 0.2681 | 0.3220 |

---

## Dataset Setup

The dataset is the **MIND-small** split (~50,000 users) from the [Microsoft News Dataset](https://msnews.github.io/).

> ⚠️ Microsoft's original Azure blob URLs are no longer publicly accessible. Download via Kaggle instead.

### Option A — Kaggle API (recommended)

1. Create a free account at [kaggle.com](https://kaggle.com)
2. Go to **Account → Settings → API → Create New Token** — this downloads `kaggle.json`
3. In your Colab/notebook, run:

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_api_key'

!kaggle datasets download -d arashnic/mind-news-dataset -p ./MIND_small/_kaggle_raw --unzip
```

### Option B — Manual download

1. Visit [kaggle.com/datasets/arashnic/mind-news-dataset](https://www.kaggle.com/datasets/arashnic/mind-news-dataset) and click **Download**
2. Unzip and place files so you have:

```
./MIND_small/train/behaviors.tsv
./MIND_small/train/news.tsv
./MIND_small/dev/behaviors.tsv
./MIND_small/dev/news.tsv
```

3. Set `MANUAL_DOWNLOAD = True` in the download cell and proceed from Section 2

### Note on the dev split

The Kaggle distribution only includes the training split. The notebook automatically creates an 80/20 train/validation split from the available data:

```python
split = int(len(beh) * 0.8)
train_beh = beh.iloc[:split]   # 125,572 impressions
dev_beh   = beh.iloc[split:]   # 31,393 impressions
```

---

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy tqdm \
            kaggle gensim sentence-transformers transformers torch
```

Or run the first cell of the notebook which installs everything automatically.

**GPU note:** The BERT, RoBERTa, and DistilBERT encoding cells benefit significantly from a GPU. On Google Colab, go to **Runtime → Change runtime type → T4 GPU** before running Section 5e–5g.

| Environment | BERT/RoBERTa/DistilBERT encoding time |
|-------------|---------------------------------------|
| Colab T4 GPU | ~2–4 min per model |
| CPU only | ~15–20 min per model |

---

## Running the Notebook

Open `HM2_News_Recommendation_v3.ipynb` in Google Colab or Jupyter and run cells top to bottom. Each section is self-contained:

```
Section 0  → Install packages & imports
Section 1  → Download & extract MIND dataset
Section 2  → Load data & exploratory analysis
Section 3  → Evaluation utilities (AUC, MRR, nDCG)
Section 4  → Model 1: Popularity Baseline
Section 5  → Models 2a–2g: All content-based encoders
             5a. TF-IDF
             5b. Word2Vec
             5c. FastText
             5d. Sentence-BERT
             5e. BERT
             5f. RoBERTa
             5g. DistilBERT
Section 6  → Model 3: User-Based Collaborative Filtering
Section 7  → Model 4: SVD / Matrix Factorization
Section 8  → Model 5: Clustering (K-Means)
Section 9  → Model 6: Hybrid (TF-IDF + SVD)
Section 10 → Results & visualizations
Section 11 → Additional analysis (PCA, alpha sweep, cold-start)
Section 12 → Qualitative sample recommendations
Section 13 → Final summary table & key takeaways
```

---

## Project Structure

```
.
├── News_Recommendation.ipynb   # Main notebook (all 12 models)
├── Report.pdf                     # 2-page project report
├── MIND_small/
│   ├── train/
│   │   ├── behaviors.tsv              # Training impression logs
│   │   ├── news.tsv                   # News article metadata
│   │   ├── entity_embedding.vec       # (optional) Entity embeddings
│   │   └── relation_embedding.vec     # (optional) Relation embeddings
│   └── dev/
│       ├── behaviors.tsv              # Validation impression logs
│       └── news.tsv                   # News article metadata
└── README.md
```

---

## Evaluation Metrics

All metrics are computed per impression and averaged across 3,000 sampled validation impressions.

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **AUC** | Area under ROC curve | Overall ranking quality |
| **MRR@10** | $\frac{1}{K}\sum_{i=1}^{K} \frac{1}{\text{rank}_i}$ | Position of first click in top-10 |
| **nDCG@5** | $\frac{\text{DCG@5}}{\text{IDCG@5}}$ | Top-5 ranking quality |
| **nDCG@10** | $\frac{\text{DCG@10}}{\text{IDCG@10}}$ | Top-10 ranking quality |

where $\text{DCG@K} = \sum_{i=1}^{K} \frac{r_i}{\log_2(i+1)}$ and IDCG is the ideal (perfect) DCG.

---

## References

- Wu et al. (2020). *MIND: A Large-scale Dataset for News Recommendation*. ACL 2020.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- Koren et al. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.

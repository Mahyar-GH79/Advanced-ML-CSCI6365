# Link Prediction on NASA GES-DISC Knowledge Graph

This project applies two link prediction methods to the [NASA GES-DISC Knowledge Graph](https://zenodo.org/record/11492533) — a graph connecting NASA Earth science datasets, instruments, platforms, science keywords, and publications.

## Overview

The NASA Goddard Earth Sciences Data and Information Services Center (GES-DISC) maintains a knowledge graph to improve dataset discoverability. Science keywords are assigned to datasets, but some links may be missing. Link prediction identifies these gaps by estimating the probability of unobserved edges in the graph.

We implement and compare two approaches:

| Method | Approach | Classifier |
|--------|----------|------------|
| **Method 1** | Node2Vec embeddings (Hadamard product edge features) | Logistic Regression |
| **Method 2** | Graph heuristics (Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment) | Random Forest |

## Dataset

**Source:** [Zenodo — NASA GES-DISC Knowledge Graph for Link Prediction (v1.0)](https://zenodo.org/record/11492533)

The dataset contains four CSV files:

- `nodes.csv` — All graph nodes with types (Dataset, Keyword, Instrument, Platform, etc.)
- `train_edges.csv` — Training edges
- `val_links.csv` — Validation edge pairs with labels
- `test_links.csv` — Test edge pairs with labels

The notebook downloads these automatically on first run.

## Project Structure

```
.
├── README.md
├── Assignment2_LinkPrediction_NASA_GES_DISC.ipynb   # Main notebook
└── data/                                            # Created at runtime
    ├── nodes.csv
    ├── train_edges.csv
    ├── val_links.csv
    └── test_links.csv
```

## Getting Started

### Prerequisites

Python 3.8+ is required.

### Setup and Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/NASA-GES-DISC-Link-Prediction.git
cd NASA-GES-DISC-Link-Prediction

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn matplotlib networkx requests node2vec

# Launch the notebook
jupyter notebook Assignment2_LinkPrediction_NASA_GES_DISC.ipynb
```

Alternatively, open the notebook directly in Google Colab — all dependencies are installed inline.

## Methods

### Method 1 — Node2Vec + Logistic Regression

1. Train Node2Vec on the graph (64-dim embeddings, walk length 20, 10 walks per node).
2. Compute edge features via the Hadamard product of source/target embeddings.
3. Train a Logistic Regression classifier on edge features.

### Method 2 — Graph Heuristics + Random Forest

1. Compute four topological features per candidate edge: Common Neighbors, Jaccard Coefficient, Adamic-Adar Index, and Preferential Attachment.
2. Train a Random Forest classifier (100 trees, max depth 10) on these features.

## Evaluation

Both methods are evaluated on the same test set using:

- Accuracy, Precision, Recall, F1 Score
- ROC-AUC and Average Precision
- Training and prediction time

The notebook includes a side-by-side comparison table and bar chart visualization.

## License

The dataset is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [Zenodo page](https://zenodo.org/record/11492533) for full terms.

## Acknowledgments

- **Dataset authors:** Armin Mehrabian and Irina Gerasimov, NASA Goddard Space Flight Center
- **Citation:** Mehrabian, A., Gerasimov, I. (2024). *NASA GES-DISC Knowledge Graph for Link Prediction* (1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.11492533

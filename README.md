# mDADGAN

Implementation of mDADGAN for miRNA–drug drug-response association prediction with a heterogeneous graph + diffusion-based perturbation + adversarial training.

## Repository Structure
mDADGAN/
├── Main/
│ └── main.py # entry point: data loading → feature extraction → graph construction → training/evaluation
├── Model/
│ └── model.py # generator & discriminator; diffusion_schedule; hetero-GNN encoder
├── Train/
│ └── train.py # training loop, negative sampling, metrics, plotting
├── hetero_graph/
│ └── mi_drug_hetero_graph.py # build DGL heterograph for (miRNA, lncRNA, Drug) and relations
├── Data/
│ ├── load_dataset.py # dataset loader
│ ├── *.xlsx # sequences, similarities, SMILES, and integrated association tables
│ └── dataset/ # (optional) cached/processed files
├── weights/ # saved checkpoints (G.pth / D.pth)
├── get_miRNA_feature.py # miRNA feature extraction (e.g., k-mer, GC content, etc.)
├── get_lncRNA_feature.py # lncRNA feature extraction (e.g., k-mer, GC content, etc.)
├── get_drug_feature.py # drug feature extraction (Morgan/MACCS/Daylight fingerprints via RDKit)
└── get_adj.py # adjacency matrix builder

## Module Overview

- **Main/main.py**: full pipeline (load dataset → build node features → construct heterograph → train/evaluate).  
- **Model/model.py**: model definitions (Generator/Discriminator), diffusion-based perturbation schedule, and heterogeneous GNN encoder.  
- **Train/train.py**: adversarial optimization, negative sampling, evaluation metrics (AUC/AUPR/ACC/Precision/Recall/F1), and training curves.  
- **hetero_graph/mi_drug_hetero_graph.py**: constructs the DGL heterograph with node features for miRNA/lncRNA/Drug.

## Environment Requirements (Exact)

This repository depends on:
- Python (recommended: 3.8+)
- PyTorch
- DGL
- RDKit (for drug fingerprints)
- Biopython (for RNA sequence utilities)
- numpy, pandas, scikit-learn, matplotlib, dill, openpyxl

**Important:** DGL must match your CUDA version (CPU or CUDA build). 


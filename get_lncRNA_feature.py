import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import math
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

def get_basic_g(n, k, letters, m='', k_letter=''):
    if m == '':
        m = n - k
        k_letter = list(letters[:n])
    if n == m + 1:
        return k_letter
    num_of_perm = []
    for perm in get_basic_g(n - 1, k, letters, m, k_letter):
        temp_letter = [i for i in k_letter]
        num_of_perm += [perm + i for i in temp_letter]
    return num_of_perm

def get_k_mer(xulie, k=3):
    basic = ['A', 'C', 'G', 'T']
    basic_group = get_basic_g(len(basic), k, basic)
    base_group_count = [dict(zip(basic_group, [0] * len(basic_group))) for _ in range(len(xulie))]
    for i in range(len(xulie)):
        for j in range(len(xulie[i]) - k + 1):
            kmer = xulie[i][j:j + k]
            if kmer in base_group_count[i]:
                base_group_count[i][kmer] += 1
    feat_f = [list(base_group_count[i].values()) for i in range(len(base_group_count))]
    feat = []
    for i in range(len(feat_f)):
        sum_f = sum(feat_f[i])
        if sum_f == 0:
            feat.append([0] * len(feat_f[0]))
        else:
            feat.append([round(feat_f[i][n] / sum_f, 4) for n in range(len(feat_f[0]))])
    return feat

def calculate_gc_content(sequence):
    return gc_fraction(sequence) if sequence else 0
def calculate_local_gc_content(sequence, window_size=5):
    gc_contents = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        gc_contents.append(gc_fraction(window))
    return sum(gc_contents) / len(gc_contents) if gc_contents else 0

def calculate_hydrogen_bonds(sequence):
    base_pairs = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    hydrogen_bonds = 0
    for i in range(len(sequence) - 1):
        if sequence[i] in base_pairs and sequence[i + 1] == base_pairs[sequence[i]]:
            if sequence[i] in ['G', 'C']:
                hydrogen_bonds += 3
            else:
                hydrogen_bonds += 2
    return hydrogen_bonds

def calculate_shannon_entropy(sequence, k=3):
    kmer_counts = Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))
    total_kmers = sum(kmer_counts.values())
    entropy = -sum((count / total_kmers) * math.log2(count / total_kmers)
                   for count in kmer_counts.values() if count > 0)
    return entropy

def get_secondary_structure_features(sequence):
    gc_content = calculate_gc_content(sequence)
    local_gc_content = calculate_local_gc_content(sequence)
    hydrogen_bonds = calculate_hydrogen_bonds(sequence)
    entropy = calculate_shannon_entropy(sequence)

    return {
        'gc_content': gc_content,
        'local_gc_content': local_gc_content,
        'hydrogen_bonds': hydrogen_bonds,
        'shannon_entropy': entropy
    }

class FeatureFusion(nn.Module):
    def __init__(self, kmer_dim, sec_dim, sim_dim, kmer_reduce=64, sec_reduce=8, sim_reduce=64, output_size=128):
        super(FeatureFusion, self).__init__()
        self.kmer_reduce = kmer_reduce
        self.sec_reduce = sec_reduce
        self.sim_reduce = sim_reduce
        self.output_size = output_size
        self.kmer_linear = nn.Linear(kmer_dim, kmer_reduce)
        self.sec_linear = nn.Linear(sec_dim, sec_reduce)
        self.sim_linear = nn.Linear(sim_dim, sim_reduce)
        self.alpha_kmer = nn.Parameter(torch.ones(kmer_reduce))
        self.alpha_sec = nn.Parameter(torch.ones(sec_reduce))
        self.alpha_sim = nn.Parameter(torch.ones(sim_reduce))
        self.transformer = nn.Linear(kmer_reduce + sec_reduce + sim_reduce, output_size)

    def forward(self, kmer_tensor, sec_tensor, sim_tensor):
        kmer_emb = self.kmer_linear(kmer_tensor)  # (N, 64)
        sec_emb = self.sec_linear(sec_tensor)    # (N, 8)
        sim_emb = self.sim_linear(sim_tensor)    # (N, 64)
        kmer_emb = kmer_emb * self.alpha_kmer
        sec_emb = sec_emb * self.alpha_sec
        sim_emb = sim_emb * self.alpha_sim
        combined_features = torch.cat([kmer_emb, sec_emb, sim_emb], dim=1)  # (N, 64+8+64=136)
        output_features = self.transformer(combined_features)  # (N, output_size)
        return output_features

def get_feature(mirna_name, seq_file_path, sim_file_path, k=3, output_size=128):
    df = pd.read_excel(seq_file_path, header=None)
    df.columns = ['miRNA', 'Sequence']
    mirna_name_seq = dict(zip(df['miRNA'], df['Sequence']))
    mirna_xulie = []
    for name in mirna_name:
        if name in mirna_name_seq:
            mirna_xulie.append(mirna_name_seq[name])
        else:
            mirna_xulie.append("")
    kmer_features = get_k_mer(mirna_xulie, k)
    kmer_tensor = torch.tensor(kmer_features, dtype=torch.float32)
    sec_features = []
    for seq in mirna_xulie:
        if seq:
            structure_features = get_secondary_structure_features(seq)
            sec_features.append([
                structure_features['gc_content'],
                structure_features['local_gc_content'],
                structure_features['hydrogen_bonds'],
                structure_features['shannon_entropy']
            ])
        else:
            sec_features.append([0, 0, 0, 0])
    sec_tensor = torch.tensor(sec_features, dtype=torch.float32)
    sim_df = pd.read_excel(sim_file_path, index_col=0)
    sim_matrix = sim_df.values
    sim_tensor = []
    for name in mirna_name:
        if name in sim_df.index:
            sim_tensor.append(sim_df.loc[name].values)
        else:
            sim_tensor.append(np.zeros(sim_matrix.shape[1]))
    sim_tensor = torch.tensor(sim_tensor, dtype=torch.float32)
    fusion_model = FeatureFusion(kmer_tensor.shape[1], sec_tensor.shape[1], sim_tensor.shape[1],
                                  kmer_reduce=64, sec_reduce=8, sim_reduce=64, output_size=output_size)
    output_features = fusion_model(kmer_tensor, sec_tensor, sim_tensor)
    print("The feature construction is completed with weighted fusion!")
    return output_features

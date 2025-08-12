import pandas as pd
import numpy as np
import torch

def normalize_name(name):
    return name.lower().strip() 

def dataset():
    mirna_name = []
    lncRNA_name = []
    drug_name = []

    input_mirna_drug = [[], []]
    input_lncRNA_drug = [[], []]
    input_mirna_lncRNA = [[], []]

    md = pd.read_excel("../Data/dataset/mDADGAN-miDrug v1.0.xlsx", header=None, names=['mirna', 'drug'])
    #md = pd.read_excel("../Data/dataset/ncDR.xlsx", header=None, names=['mirna', 'drug'])
    # md = pd.read_excel("../Data/dataset/RNAInter.xlsx", header=None, names=['mirna', 'drug'])
    # md = pd.read_excel("../Data/dataset/SM2miR3.xlsx", header=None, names=['mirna', 'drug'])

    md['mirna'] = md['mirna'].str.lower()
    md['drug'] = md['drug'].str.lower()
    for _, row in md.iterrows():
        mirna = row['mirna']
        drug = row['drug']
        if mirna not in mirna_name:
            mirna_name.append(mirna)
        if drug not in drug_name:
            drug_name.append(drug)
        input_mirna_drug[0].append(mirna_name.index(mirna))
        input_mirna_drug[1].append(drug_name.index(drug))

    td = pd.read_excel("../Data/dataset/lncRNA-drug.xlsx", header=None, names=['lncRNA', 'drug'])
    td['lncRNA'] = td['lncRNA'].str.lower()
    td['drug'] = td['drug'].str.lower()
    for _, row in td.iterrows():
        lncRNA = row['lncRNA']
        drug = row['drug']
        if lncRNA not in lncRNA_name:
            lncRNA_name.append(lncRNA)
        if drug not in drug_name:
            drug_name.append(drug)
        input_lncRNA_drug[0].append(lncRNA_name.index(lncRNA))
        input_lncRNA_drug[1].append(drug_name.index(drug))

    mt = pd.read_excel("../Data/dataset/lncRNA-miRNA.xlsx", header=None, names=['lncRNA','mirna'])
    mt['mirna'] = mt['mirna'].str.lower()
    mt['lncRNA'] = mt['lncRNA'].str.lower()
    for _, row in mt.iterrows():
        mirna = row['mirna']
        lncRNA = row['lncRNA']
        if mirna not in mirna_name:
            mirna_name.append(mirna)
        if lncRNA not in lncRNA_name:
            lncRNA_name.append(lncRNA)
        input_mirna_lncRNA[0].append(mirna_name.index(mirna))
        input_mirna_lncRNA[1].append(lncRNA_name.index(lncRNA))

    mirna_num = len(mirna_name)
    lncRNA_num = len(lncRNA_name)
    drug_num = len(drug_name)

    print("\ndataset7 loading completed!")
    return mirna_num, lncRNA_num, drug_num, input_mirna_drug, input_lncRNA_drug, input_mirna_lncRNA, mirna_name, lncRNA_name, drug_name

if __name__ == '__main__':
    dataset()

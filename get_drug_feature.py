import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.Chem import AllChem
import torch.nn as nn

def get_fingerprint(smiles, radius=2, fp_size=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size)
        return torch.tensor(list(morgan_fp), dtype=torch.float32)
    else:
        return torch.zeros(fp_size)

def get_MACCS_keys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        return torch.tensor(list(maccs_fp), dtype=torch.float32)
    else:
        return torch.zeros(167)

def get_daylight_fingerprint(smiles, fp_size=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        daylight_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
        return torch.tensor(list(daylight_fp), dtype=torch.float32)
    else:
        return torch.zeros(fp_size)

def extract_drug_features(drug_name, file_path):
    df = pd.read_excel(file_path)
    df.columns = ['Drug', 'SMILES']
    drug_name_smiles = dict(zip(df['Drug'], df['SMILES']))
    morgan_features = []
    maccs_features = []
    daylight_features = []

    for name in drug_name:
        smiles = drug_name_smiles.get(name, "")
        if smiles:
            morgan_fp = get_fingerprint(smiles)
            maccs_fp = get_MACCS_keys(smiles)
            daylight_fp = get_daylight_fingerprint(smiles)
            morgan_features.append(morgan_fp)
            maccs_features.append(maccs_fp)
            daylight_features.append(daylight_fp)
        else:
            morgan_features.append(torch.zeros(2048))
            maccs_features.append(torch.zeros(167))
            daylight_features.append(torch.zeros(2048))
    return torch.stack(morgan_features), torch.stack(maccs_features), torch.stack(daylight_features)


class FeatureFusionModel(nn.Module):
    def __init__(self, morgan_size, maccs_size, daylight_size, output_size=128):
        super(FeatureFusionModel, self).__init__()
        self.morgan_fc = nn.Linear(morgan_size, 512)
        self.maccs_fc = nn.Linear(maccs_size, 512)
        self.daylight_fc = nn.Linear(daylight_size, 512)
        self.attention = nn.Linear(512 * 3, 1)
        self.fusion_fc = nn.Linear(512 * 3, output_size)
        self.relu = nn.ReLU()
    def forward(self, morgan_fp, maccs_fp, daylight_fp):
        morgan_out = self.relu(self.morgan_fc(morgan_fp))
        maccs_out = self.relu(self.maccs_fc(maccs_fp))
        daylight_out = self.relu(self.daylight_fc(daylight_fp))
        combined = torch.cat([morgan_out, maccs_out, daylight_out], dim=1)
        attention_weights = torch.softmax(self.attention(combined), dim=1)
        weighted_combined = combined * attention_weights
        output = self.fusion_fc(weighted_combined)
        return output
model = FeatureFusionModel(morgan_size=2048, maccs_size=167, daylight_size=2048, output_size=128)



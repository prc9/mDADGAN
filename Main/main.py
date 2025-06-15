import torch
import random
import warnings
import numpy as np
from torch.utils.hipify.hipify_python import mapping
from Data import dataset
from Model import model
from Train import train
from sklearn.model_selection import train_test_split
from Data.load_dataset import dataset
import get_miRNA_feature
import get_lncRNA_feature
import get_drug_feature
from hetero_graph.mi_drug_hetero_graph import get_hetero_graph
import dill
from get_adj import adj
from Model import model

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): print('gpu ok')
    epochs = 150
    pro_ZR = 50
    pro_PM = 50
    alpha = 0.1

    feat_shape = 128
    out_feat = 128

    G_step = 5
    D_step = 2
    batchSize = 32

    G_PATH = '../weights/G.pth'
    D_PATH = '../weights/D.pth'

    # 加载数据集（这里保持原数据集不变）
    mirna_num, lncRNA_num, drug_num, mirna_drug, lncRNA_drug, mirna_lncRNA, mirna_name, lncRNA_name, drug_name=dataset()

    # 获取特征（保持原特征处理不变）
    mirna_feat = get_miRNA_feature.get_feature(mirna_name, '../Data/miRNA_sequence.xlsx','../Data/normalized.xlsx')
    lncRNA_feat = get_lncRNA_feature.get_feature(lncRNA_name, '../Data/lncRNA_sequence.xlsx', '../Data/lncRNA_lncRNA_similarity.xlsx')
    morgan_features, maccs_features, daylight_features = get_drug_feature.extract_drug_features(drug_name, '../Data/drug smile.xlsx')
    drug_feat = get_drug_feature.fused_features = get_drug_feature.model(morgan_features, maccs_features, daylight_features)

    print("\nData Details:")
    print("miRNA:{}  lncRNA:{}  drug:{}".format(mirna_num, lncRNA_num, drug_num))
    print("miRNA-drug:{}  lncRNA-drug:{}  miRNA-lncRNA:{}".format(len(mirna_drug[0]), len(lncRNA_drug[0]),
                                                                        len(mirna_lncRNA[0])))
    print("Sparsity of miRNA-drug associated data: {:.6f}\n".format(
        len(mirna_drug[0]) / (mirna_num * drug_num)))
    edge_data = np.array([[mirna_drug[0][i], mirna_drug[1][i]] for i in range(len(mirna_drug[0]))])
    total_pairs = mirna_num * drug_num
    pos_num = len(edge_data)
    neg_num = total_pairs - pos_num
    pos_ratio = pos_num / total_pairs
    neg_ratio = neg_num / total_pairs
    all_indices = np.arange(len(edge_data))
    train_index, test_index = train_test_split(all_indices, test_size=0.2, shuffle=True)

    train_negative = []
    for _ in range(len(train_index)):
        while True:
            row = random.randint(0, mirna_num - 1)
            col = random.randint(0, drug_num - 1)
            if not any((edge_data[:, 0] == row) & (edge_data[:, 1] == col)) and [row, col] not in train_negative:
                train_negative.append([row, col])
                break

    test_negative = []
    for _ in range(len(test_index)):
        while True:
            row = random.randint(0, mirna_num - 1)
            col = random.randint(0, drug_num - 1)
            if not any((edge_data[:, 0] == row) & (edge_data[:, 1] == col)) and [row, col] not in train_negative + test_negative:
                test_negative.append([row, col])
                break

    train_mirna_20 = random.sample(train_index.tolist(), int(len(train_index) * 0.25))
    train_mirna_60 = [i for i in train_index if i not in train_mirna_20]

    input_net = [[], []]
    for i in train_mirna_60:
        input_net[0].append(edge_data[i][0])
        input_net[1].append(edge_data[i][1])
    for neg in train_negative:
        input_net[0].append(neg[0])
        input_net[1].append(neg[1])

    true_input_net = [[], []]
    for i in train_index:
        true_input_net[0].append(edge_data[i][0])
        true_input_net[1].append(edge_data[i][1])

    test_input_net = [[], []]
    for i in test_index:
        test_input_net[0].append(edge_data[i][0])
        test_input_net[1].append(edge_data[i][1])
    for neg in test_negative:
        test_input_net[0].append(neg[0])
        test_input_net[1].append(neg[1])

    G = model.generator(drug_num, feat_shape, out_feat).to(device)
    D = model.discriminator(drug_num, feat_shape, out_feat).to(device)
    Noise_MLDN, Noise_MLDN_h, True_MLDN, True_MLDN_h = get_hetero_graph(
        input_net, true_input_net, mirna_lncRNA, lncRNA_drug,
        mirna_feat, lncRNA_feat, drug_feat
    )

    # 训练并测试
    auc = train.main(
        mirna_num, drug_num, epochs, pro_ZR, pro_PM, alpha, batchSize,
        input_net, true_input_net, test_input_net, test_negative,
        Noise_MLDN, Noise_MLDN_h, True_MLDN, True_MLDN_h,
        G, D, G_step, D_step, G_PATH, D_PATH
    )

    print(f'\nFinal AUC: {auc:.4f}')

    G.load_state_dict(torch.load(G_PATH, map_location=device))
    G.to(device)
    G.eval()

import copy
import dill
import torch
import random
import numpy as np
import torch.nn as nn
from get_adj import adj
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def select_negative_items(Data, num_pm, num_zr, disease_num):
    if isinstance(Data, torch.Tensor):
        data = Data.cpu().numpy()
    else:
        data = np.array(Data)
    n_items_pm = np.zeros_like(data)
    n_items_zr = np.zeros_like(data)
    for i in range(data.shape[0]):
        p_items = np.where(data[i] != 0)[0]
        all_item_index = random.sample(range(data.shape[1]), disease_num)
        for j in range(p_items.shape[0]):
            all_item_index.remove(list(p_items)[j])
        random.shuffle(all_item_index)
        n_item_index_pm = all_item_index[0: num_pm]
        n_item_index_zr = all_item_index[num_pm: (num_pm + num_zr)]
        n_items_pm[i][n_item_index_pm] = 1
        n_items_zr[i][n_item_index_zr] = 1
    return n_items_pm, n_items_zr


def main(mirna_num, drug_num, epochCount, pro_ZR, pro_PM, alpha, batchSize,
         input_net, true_input_net, test_input_net, test_negative,
         Noise_MLDN, Noise_MLDN_h, True_MLDN, True_MLDN_h,
         G, D, G_step, D_step, G_PATH, D_PATH
         ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_auc = 0
    t = 0
    auc_scores = []
    aupr_scores = []

    regularization = nn.MSELoss()
    criterion = nn.BCELoss()

    #0.0015 0.0001  0.025
    d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.0001
                                      )

    Noise_MDA = adj(mirna_num, drug_num, input_net).to(device)
    True_MDA = adj(mirna_num, drug_num, true_input_net).to(device)
    Test_Adj = adj(mirna_num, drug_num, test_input_net).to(device)

    for epoch in range(epochCount):
        for step in range(G_step):
            leftIndex = random.randint(0, mirna_num - batchSize - 1)
            realData = Variable(copy.deepcopy(True_MDA[leftIndex:leftIndex + batchSize]))
            noiseData = Variable(copy.deepcopy(Noise_MDA[leftIndex:leftIndex + batchSize]))
            e_i = Variable(copy.deepcopy(Noise_MDA[leftIndex:leftIndex + batchSize]))
            n_items_pm, n_items_zr = select_negative_items(noiseData, pro_PM, pro_ZR, drug_num)
            k_i_zp = Variable(torch.tensor(n_items_pm + n_items_zr))
            realData_zp = Variable(torch.ones_like(realData)) * e_i + Variable(torch.zeros_like(realData)) * k_i_zp
            fake_embeding, r_i = G(Noise_MLDN, Noise_MLDN_h, noiseData, batchSize, leftIndex)
            pred_matrix = r_i * (e_i + k_i_zp)
            fakeData_result = D(pred_matrix, fake_embeding)
            g_loss = -torch.mean(fakeData_result) + alpha * regularization(pred_matrix, realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()


        for step in range(D_step):
            leftIndex = random.randint(1, mirna_num - batchSize - 1)
            realData = Variable(copy.deepcopy(True_MDA[leftIndex:leftIndex + batchSize]))
            noise_Data = Variable(copy.deepcopy(Noise_MDA[leftIndex:leftIndex + batchSize]))

            e_i = Variable(copy.deepcopy(Noise_MDA[leftIndex:leftIndex + batchSize]))
            n_items_pm, _ = select_negative_items(noise_Data, pro_PM, pro_ZR, drug_num)
            k_i = Variable(torch.tensor(n_items_pm))

            fake_embeding, r_i = G(Noise_MLDN, Noise_MLDN_h, noise_Data, batchSize, leftIndex)
            pred_matrix = r_i * (e_i + k_i)
            fakeData_result = D(pred_matrix, fake_embeding)

            true_embeding, _ = G(True_MLDN, True_MLDN_h, realData, batchSize, leftIndex)
            realData_result = D(realData, true_embeding)

            cost_fn = 10
            cost_fp = 1
            d_loss = -cost_fn * torch.mean(realData_result) + cost_fp * torch.mean(fakeData_result)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        label = []
        pred = []
        for testUser in range(len(Test_Adj)):
            data = Variable(copy.deepcopy(Noise_MDA[testUser:testUser + 1]))
            _, predData = G(Noise_MLDN, Noise_MLDN_h, data, 1, testUser)
            pred_i = predData[0].cpu().tolist()
            test_i = Test_Adj[testUser].cpu().tolist()
            for i in range(len(test_i)):
                if test_i[i] == 1 and [testUser, i] in test_negative:
                    label.append(0)
                    pred.append(pred_i[i])
                if test_i[i] == 1 and [testUser, i] not in test_negative:
                    label.append(1)
                    pred.append(pred_i[i])

  # Compute AUC and AUPR
        auc = roc_auc_score(label, pred)
        p, r, thresholds = precision_recall_curve(label, pred)
        aupr = metrics.auc(r, p)

        # Compute additional metrics
        pred_binary = [1 if x >= 0.5 else 0 for x in pred]  # Assuming threshold of 0.5 for binary classification
        accuracy = accuracy_score(label, pred_binary)
        precision = precision_score(label, pred_binary)
        recall = recall_score(label, pred_binary)  # Sensitivity
        f1 = f1_score(label, pred_binary)

        # Compute confusion matrix to get specificity
        tn, fp, fn, tp = confusion_matrix(label, pred_binary).ravel()
        specificity = tn / (tn + fp)

        # Print metrics
        print(f'Epoch[{epoch}/{epochCount}], AUC: {auc:.4f}, AUPR: {aupr:.4f}, '
              f'ACC: {accuracy:.4f}, Precision: {precision:.4f}, Sensitivity (Recall): {recall:.4f}, '
              f'Specificity: {specificity:.4f}, F1 Score: {f1:.4f}')

        auc_scores.append(auc)
        aupr_scores.append(aupr)

        if auc > best_auc:
            best_auc = auc
            torch.save(G.state_dict(), G_PATH)
            torch.save(D.state_dict(), D_PATH)

        if auc < best_auc:
            t -= 1
        else:
            t = 0
        if t <= -10: break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(auc_scores) + 1), auc_scores, 'b-', marker='o')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(aupr_scores) + 1), aupr_scores, 'r-', marker='o')
    plt.title('AUPR over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUPR')

    plt.show()

    return best_auc



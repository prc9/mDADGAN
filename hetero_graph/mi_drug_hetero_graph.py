import dgl
import torch

def get_hetero_graph(input_net, true_input_net, mirna_lncRNA, lncRNA_drug, mirna_feat, lncRNA_feat, drug_feat):
    Noise_MLDN_data = {
        ('MiRNA', 'MvsD', 'Drug'): (torch.tensor(input_net[0]), torch.tensor(input_net[1])),
        ('Drug', 'DvsM', 'MiRNA'): (torch.tensor(input_net[1]), torch.tensor(input_net[0])),
        ('MiRNA', 'MvsL', 'LncRNA'): (torch.tensor(mirna_lncRNA[0]), torch.tensor(mirna_lncRNA[1])),
        ('LncRNA', 'LvsM', 'MiRNA'): (torch.tensor(mirna_lncRNA[1]), torch.tensor(mirna_lncRNA[0])),
        ('LncRNA', 'LvsD', 'Drug'): (torch.tensor(lncRNA_drug[0]), torch.tensor(lncRNA_drug[1])),
        ('Drug', 'DvsL', 'LncRNA'): (torch.tensor(lncRNA_drug[1]), torch.tensor(lncRNA_drug[0]))
    }

    Noise_MLDN = dgl.heterograph(Noise_MLDN_data)
    Noise_MLDN.nodes['MiRNA'].data['feat'] = mirna_feat
    Noise_MLDN.nodes['LncRNA'].data['feat'] = lncRNA_feat
    Noise_MLDN.nodes['Drug'].data['feat'] = drug_feat
    Noise_MLDN_h = {'MiRNA': Noise_MLDN.nodes['MiRNA'].data['feat'],
                    'LncRNA': Noise_MLDN.nodes['LncRNA'].data['feat'],
                    'Drug': Noise_MLDN.nodes['Drug'].data['feat']}

    True_MTDN_data = {
        ('MiRNA', 'MvsD', 'Drug'): (torch.tensor(true_input_net[0]), torch.tensor(true_input_net[1])),
        ('Drug', 'DvsM', 'MiRNA'): (torch.tensor(true_input_net[1]), torch.tensor(true_input_net[0])),
        ('MiRNA', 'MvsL', 'LncRNA'): (torch.tensor(mirna_lncRNA[0]), torch.tensor(mirna_lncRNA[1])),
        ('LncRNA', 'LvsM', 'MiRNA'): (torch.tensor(mirna_lncRNA[1]), torch.tensor(mirna_lncRNA[0])),
        ('LncRNA', 'LvsD', 'Drug'): (torch.tensor(lncRNA_drug[0]), torch.tensor(lncRNA_drug[1])),
        ('Drug', 'DvsL', 'LncRNA'): (torch.tensor(lncRNA_drug[1]), torch.tensor(lncRNA_drug[0]))
    }

    True_MLDN = dgl.heterograph(True_MTDN_data)
    True_MLDN.nodes['MiRNA'].data['feat'] = mirna_feat
    True_MLDN.nodes['LncRNA'].data['feat'] = lncRNA_feat
    True_MLDN.nodes['Drug'].data['feat'] = drug_feat
    True_MLDN_h = {'MiRNA': True_MLDN.nodes['MiRNA'].data['feat'],
                   'LncRNA': True_MLDN.nodes['LncRNA'].data['feat'],
                   'Drug': True_MLDN.nodes['Drug'].data['feat']}

    return Noise_MLDN, Noise_MLDN_h, True_MLDN, True_MLDN_h

import torch
import torch.nn.functional as F
from torch import device
import copy
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from torch.autograd import Variable

def diffusion_schedule(timesteps, device=torch.device("cpu")):
    min_rate = 0.0001
    max_rate = 0.02
    betas = torch.linspace(min_rate, max_rate, timesteps, dtype=torch.float32, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1 - alpha_bars)
    return noise_rates, signal_rates


class generator(nn.Module):
    def __init__(self, itemCount, feat_shape, out_feat_shape, timesteps=100):
        super(generator, self).__init__()
        self.timesteps = timesteps
        self.itemCount = itemCount
        self.feat_shape = feat_shape
        self.out_feat_shape = out_feat_shape

        self.HeteroConv1 = dglnn.HeteroGraphConv({
            'MvsD': dglnn.SAGEConv(self.feat_shape, 64, aggregator_type='pool'),
            'DvsM': dglnn.SAGEConv(self.feat_shape, 64, aggregator_type='pool'),
            'MvsL': dglnn.SAGEConv(self.feat_shape, 64, aggregator_type='pool'),
            'LvsM': dglnn.SAGEConv(self.feat_shape, 64, aggregator_type='pool'),
            'LvsD': dglnn.SAGEConv(self.feat_shape, 64, aggregator_type='pool'),
            'DvsL': dglnn.SAGEConv(self.feat_shape, 64, aggregator_type='pool')
        }, aggregate='sum')

        self.HeteroConv2 = dglnn.HeteroGraphConv({
            'MvsD': dglnn.SAGEConv(64, 64, aggregator_type='pool'),
            'DvsM': dglnn.SAGEConv(64, 64, aggregator_type='pool'),
            'MvsL': dglnn.SAGEConv(64, 64, aggregator_type='pool'),
            'LvsM': dglnn.SAGEConv(64, 64, aggregator_type='pool'),
            'LvsD': dglnn.SAGEConv(64, 64, aggregator_type='pool'),
            'DvsL': dglnn.SAGEConv(64, 64, aggregator_type='pool')
        }, aggregate='sum')

        self.HeteroConv3 = dglnn.HeteroGraphConv({
            'MvsD': dglnn.SAGEConv(64, self.out_feat_shape, aggregator_type='pool'),
            'DvsM': dglnn.SAGEConv(64, self.out_feat_shape, aggregator_type='pool'),
            'MvsL': dglnn.SAGEConv(64, self.out_feat_shape, aggregator_type='pool'),
            'LvsM': dglnn.SAGEConv(64, self.out_feat_shape, aggregator_type='pool'),
            'LvsD': dglnn.SAGEConv(64, self.out_feat_shape, aggregator_type='pool'),
            'DvsL': dglnn.SAGEConv(64, self.out_feat_shape, aggregator_type='pool')
        }, aggregate='sum')

        self.f1 = nn.Linear(self.itemCount + self.out_feat_shape, 256)
        self.a1 = nn.ReLU(True)
        self.f2 = nn.Linear(256, 512)
        self.a2 = nn.ReLU(True)
        self.f3 = nn.Linear(512, 1024)
        self.a3 = nn.ReLU(True)
        self.f4 = nn.Linear(1024, itemCount)
        self.a4 = nn.Sigmoid()

    def forward(self, g, h, Adj, size, leftIndex):
        h1 = self.HeteroConv1(g, h)
        h2 = self.HeteroConv2(g, h1)
        h3 = self.HeteroConv3(g, h2)
        mirna_feat = h3.get('MiRNA')

        mirna_feat = torch.nn.functional.normalize(mirna_feat, p=1, dim=1)
        device = mirna_feat.device
        noise_rates, signal_rates = diffusion_schedule(self.timesteps, device)
        t = torch.randint(0, self.timesteps, (1,), device=device).item()
        noise = torch.randn_like(mirna_feat)
        noisy_embedding = signal_rates[t] * mirna_feat + noise_rates[t] * noise
        fake_embedding = noisy_embedding[leftIndex:leftIndex + size].detach().clone()

        M = torch.cat([Adj, fake_embedding], 1)
        x = self.f1(M)
        x = self.a1(x)
        x = self.f2(x)
        x = self.a2(x)
        x = self.f3(x)
        x = self.a3(x)
        x = self.f4(x)
        x = self.a4(x)
        return fake_embedding, x

class discriminator(nn.Module):
    def __init__(self, itemCount, feat_shape, out_feat_shape):
        super(discriminator, self).__init__()

        self.itemCount = itemCount
        self.feat_shape = feat_shape
        self.out_feat_shape = out_feat_shape

        self.f1 = nn.Linear(self.itemCount + self.out_feat_shape, 1024)
        self.a1 = nn.ReLU(True)
        self.f2 = nn.Linear(1024, 128)
        self.a2 = nn.ReLU(True)
        self.f3 = nn.Linear(128, 16)
        self.a3 = nn.ReLU(True)
        self.f4 = nn.Linear(16, 1)

    def forward(self, Adj, embedding):
        x = torch.cat((Adj, embedding), 1)
        x = self.f1(x)
        x = self.a1(x)
        x = self.f2(x)
        x = self.a2(x)
        x = self.f3(x)
        x = self.a3(x)
        x = self.f4(x)
        return x

    def gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, device=real_data.device)
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

        pred = self.forward(interpolated, torch.zeros_like(interpolated))

        grads = torch.autograd.grad(outputs=pred, inputs=interpolated,
                                    grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True)[0]

        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

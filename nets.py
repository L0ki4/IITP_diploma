import numpy as np
from torchvision.models import resnet34
from torch import nn
import torch
import torch.nn.functional as F

from dpipe.torch import model


class EmbeddingExtractor(nn.Module):
    def __init__(self, emb_dim=64):
        super(EmbeddingExtractor, self).__init__()
        model = resnet34()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = nn.Sequential(*list(model.children())[:-1])

        self.emb_dim = emb_dim
        if emb_dim is not None:
            self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        if x.dim() != 4:
            x = x.unsqueeze(0)

        slice_amount = x.shape[1]

        x_input = x.reshape(-1, 1, *x.shape[2:]).float()
        out = self.resnet(x_input).view(x_input.shape[0], -1)

        if self.emb_dim is not None:
            out = self.fc(out)  # relu?
        out_norm = F.normalize(out, p=2, dim=-1)  # across descriptor dim

        return out_norm.reshape(x.shape[0], slice_amount, out_norm.shape[-1])


class ResNet_agg(nn.Module):
    def __init__(self, n_classes, emb_dim=64):
        super(ResNet_agg, self).__init__()
        self.embedder = EmbeddingExtractor(emb_dim)
        if emb_dim is None:
            emb_dim = 512

        self.fc2 = nn.Linear(emb_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, meta):
        slice_mask = meta
        embedding = self.embedder(x)

        output_aggregated = (embedding * slice_mask.unsqueeze(-1)).sum(1)

        agg_norm = F.normalize(output_aggregated, p=2, dim=-1)  # relu?

        return self.fc2(agg_norm)


# https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class GhostVLAD(nn.Module):
    def __init__(self, num_clusters=64, num_g_cl=0, dim=128, out_dim=128, alpha=1.0):
        super(GhostVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.num_g_cl = num_g_cl
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.conv = nn.Conv2d(dim, num_clusters + num_g_cl, kernel_size=(1, 1), bias=True)

        init = self._init_params()
        self.centroids = nn.Parameter(init[:self.num_clusters].detach(), requires_grad=True)

        if out_dim is not None:
            self.fc = nn.Linear(num_clusters * dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.relu = nn.ReLU()

    def _init_params(self):
        init = F.normalize(torch.randn((self.num_clusters + self.num_g_cl, self.dim)), dim=-1)
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * init).unsqueeze(-1).unsqueeze(-1), requires_grad=True
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * init.norm(p=2, dim=1), requires_grad=True
        )

        return init

    def forward(self, x, slice_mask):
        # x shape batch, N, D_f
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # x = F.normalize(x, p=2, dim=-1)  # across descriptor dim

        soft_assign = self.conv(x.permute(0, 2, 1).unsqueeze(3).repeat(1, 1, 1, 2))
        soft_assign = soft_assign[:, :, :, 0]
        soft_assign = soft_assign.permute(0, 2, 1)
        if self.num_g_cl > 0:
            soft_assign = soft_assign[:, :, :-self.num_g_cl]

        residual = x.unsqueeze(2) - self.centroids.unsqueeze(0).unsqueeze(0)

        soft_assign = soft_assign * slice_mask.unsqueeze(-1).detach()
        vlad = (residual * soft_assign.unsqueeze(-1)).sum(1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        if self.out_dim is not None:
            vlad = self.bn(self.fc(vlad))  # relu?

        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class GhostVlad(nn.Module):
    def __init__(self, n_classes, num_clusters, num_g_cl, alpha, emb_dim, out_dim):
        super(GhostVlad, self).__init__()
        self.embedder = EmbeddingExtractor(emb_dim)

        if emb_dim is None:
            emb_dim = 512

        self.vlad = GhostVLAD(num_clusters=num_clusters, num_g_cl=num_g_cl, alpha=alpha, dim=emb_dim,
                              out_dim=out_dim)

        if out_dim is None:
            out_dim = num_clusters * emb_dim
        self.fc2 = nn.Linear(out_dim, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, meta):
        slice_mask = meta

        embedding = self.embedder(x)  # relu?

        vlad_out = self.vlad(embedding, slice_mask)  # relu?

        return self.fc2(vlad_out)


def model_predict(x, net=None):
    output = model.inference_step(*x, architecture=net)
    return np.argmax(output, 1)

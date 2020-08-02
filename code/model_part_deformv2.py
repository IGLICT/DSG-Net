"""
    This file defines part defromation gradient VAE/AE model.
"""

import torch
import torch.nn as nn
import GraphConv as GC


class PartFeatSampler(nn.Module):

    def __init__(self, in_size, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic
        middle_dim = 4096

        # self.mlp1 = nn.Linear(in_size, feature_size, bias = False)
        self.mlp2mu = nn.Linear(in_size, feature_size, bias = False)
        self.mlp2var = nn.Linear(in_size, feature_size, bias = False)
        # self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, x):
        # x = self.linear(x)
        # encode = nn.functional.leaky_relu(self.mlp1(x))
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            # print(std.shape)
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class PartDeformEncoder2(nn.Module):

    def __init__(self, num_point, feat_len, edge_index = None, probabilistic=False, structure_feat_len = 128, bn = False):
        super(PartDeformEncoder2, self).__init__()
        self.probabilistic = probabilistic
        self.bn = bn
        self.edge_index = edge_index
        self.structure_feat_len = structure_feat_len

        self.conv1_logr = GC.GCNConv(3, 3, edge_index)
        self.conv1_s = GC.GCNConv(6, 6, edge_index)
        self.conv4_s = GC.GCNConv(6, 6, edge_index)

        if self.bn:
            self.bn1 = torch.nn.InstanceNorm1d(9)
            self.bn2 = torch.nn.InstanceNorm1d(9)
            self.bn3 = torch.nn.InstanceNorm1d(9)
            # self.bn3 = InstanceNorm(9)
        self.mlp1_logr = nn.Linear(num_point*3, feat_len)
        self.mlp1_s = nn.Linear(num_point*6, feat_len)
        self.sampler = PartFeatSampler(in_size = feat_len * 2, feature_size=feat_len, probabilistic=probabilistic)

    def forward(self, featurein):
        feature = featurein
        feature_logr = featurein[:,:,:3]/4.0
        feature_s = featurein[:,:,3:9]/50.0
        self.vertex_num = feature.shape[1]
        # print(self.vertex_num)
        if self.bn:
            net = nn.functional.leaky_relu(self.bn1(self.conv1(feature).transpose(2, 1)).transpose(2, 1), negative_slope=0.2)
            # net = nn.functional.leaky_relu(self.bn2(self.conv2(net).transpose(2, 1)).transpose(2, 1), negative_slope=0.2)
            # net = nn.functional.leaky_relu(self.bn3(self.conv3(net).transpose(2, 1)).transpose(2, 1), negative_slope=0.2)
            net = nn.functional.leaky_relu(self.conv4(net), negative_slope=0.2)
        else:

            net_logr = torch.tanh(self.conv1_logr(feature_logr))
            net_s = torch.tanh(self.conv1_s(feature_s))
            net_s = torch.tanh(self.conv4_s(net_s))
            # net = torch.tanh(self.conv3(net, edge_index))
        # print(net.shape)
        net_logr = torch.tanh(self.mlp1_logr(net_logr.view(-1, self.vertex_num * 3)))
        net_s = torch.tanh(self.mlp1_s(net_s.view(-1, self.vertex_num * 6)))
        net = torch.cat([net_logr, net_s], dim = 1)
        # net = net.contiguous().view(-1, self.vertex_num * (9+self.structure_feat_len))
        # net = self.mlp1(net)

        net = self.sampler(net)

        return net


class PartDeformDecoder2(nn.Module):

    def __init__(self, feat_len, num_point, edge_index= None, structure_feat_len = 128, bn = False):
        super(PartDeformDecoder2, self).__init__()
        self.num_point = num_point
        self.bn = bn
        self.feat_len = feat_len
        middle_dim = 2048

        self.mlp2 = nn.Linear(feat_len+structure_feat_len, feat_len*2, bias = False)
        self.mlp1_logr = nn.Linear(feat_len, self.num_point * 3, bias = False)
        self.mlp1_s = nn.Linear(feat_len, self.num_point * 6, bias = False)
        self.conv1_logr = GC.GCNConv(3, 3, edge_index)
        self.conv1_s = GC.GCNConv(6, 6, edge_index)
        # self.conv2 = GC.GCNConv(9, 9, edge_index)
        # self.conv3 = GC.GCNConv(9, 9, edge_index)
        # self.conv4_logr = GC.GCNConv(3, 3, edge_index)
        self.conv4_s = GC.GCNConv(6, 6, edge_index)

        if bn:
            self.bn1 = torch.nn.InstanceNorm1d(9)
            self.bn2 = torch.nn.InstanceNorm1d(9)
            self.bn3 = torch.nn.InstanceNorm1d(9)
            # self.bn2 = instance_norm(9)

        self.L2Loss = nn.L1Loss(reduction='mean')

    def forward(self, net):
        # printprint(self.num_point)
        net = torch.tanh(self.mlp2(net))
        net_logr = torch.tanh(self.mlp1_logr(net[:, :self.feat_len]).view(-1, self.num_point, 3))
        net_s = torch.tanh(self.mlp1_s(net[:, self.feat_len:2*self.feat_len]).view(-1, self.num_point, 6))
        # net_logr = net[:,:,:3]
        # net_s = net[:,:,3:]
        if self.bn:
            net = nn.functional.leaky_relu(self.bn1(self.conv1(net).transpose(2,1)).transpose(2,1), negative_slope=0.2)
            # net = nn.functional.leaky_relu(self.bn2(self.conv2(net).transpose(2,1)).transpose(2,1), negative_slope=0.2)
            # net = nn.functional.leaky_relu(self.bn3(self.conv3(net).transpose(2,1)).transpose(2,1), negative_slope=0.2)
        else:
            net_logr = self.conv1_logr(net_logr)
            net_s = torch.tanh(self.conv1_s(net_s))
            # net = nn.functional.leaky_relu(self.conv2(net), negative_slope=0.2)
            # net = nn.functional.leaky_relu(self.conv3(net), negative_slope=0.2)

        # net_logr = self.conv4_logr(net_logr)
        net_s = self.conv4_s(net_s)
        # net = torch.tanh(self.conv4(net))*2.0
        net = torch.cat([net_logr*4.0, net_s*50.0], dim = 2)

        return net

    def loss(self, pred, gt):
        avg_loss = self.L2Loss(pred, gt) * 1000000

        return avg_loss

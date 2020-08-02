"""
    This file defines the box-represented shape VAE/AE model (no edge). with skip link operation
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from chamfer_distance import ChamferDistance
from utils import linear_assignment, load_pts, transform_pc_batch, get_surface_reweighting_batch
from PyTorchEMD.emd import earth_mover_distance
from feature2vertex import Feature2Vertex_pytorch
from model_part_deformv2 import PartDeformEncoder2, PartDeformDecoder2


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = F.leaky_relu(self.mlp1(x))
        mu = x + self.mlp2mu(encode)

        if self.probabilistic:
            logvar = x + self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class BoxEncoder(nn.Module):

    def __init__(self, in_size, feature_size, hidden_size):
        super(BoxEncoder, self).__init__()
        self.mlp_skip = nn.Linear(in_size, feature_size)
        self.mlp1 = nn.Linear(in_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, box_input):
        net = F.leaky_relu(self.mlp1(box_input))
        net = F.leaky_relu(self.mlp_skip(box_input) + self.mlp2(net))
        return net


class SymmetricChildEncoder(nn.Module):

    def __init__(self, feature_size_struct, feature_size_geo, hidden_size, symmetric_type, max_part_per_parent, Tree):
        super(SymmetricChildEncoder, self).__init__()

        print(f'Using Symmetric Type: {symmetric_type}')
        self.struct_symmetric_type = symmetric_type
        self.geo_symmetric_type = symmetric_type
        self.max_part_per_parent = max_part_per_parent

        self.child_op = nn.Linear(Tree.num_sem + self.max_part_per_parent + feature_size_struct, hidden_size)
        self.second = nn.Linear(hidden_size, feature_size_struct)
        self.second_norm = nn.GroupNorm(num_groups=min(32, feature_size_struct//8), num_channels=feature_size_struct)
        self.skip_op = nn.Linear(feature_size_struct + self.max_part_per_parent + Tree.num_sem, feature_size_struct)

        self.child_op_geo = nn.Linear(feature_size_geo, hidden_size)
        self.second_geo = nn.Linear(hidden_size, feature_size_geo)
        self.second_norm_geo = nn.GroupNorm(num_groups=min(32, feature_size_geo//8), num_channels=feature_size_geo)
        self.skip_op_geo = nn.Linear(feature_size_geo, feature_size_geo)

    def geo_forward(self, child_geo_feats, child_exists):
        batch_size = child_geo_feats.shape[0]
        max_childs = child_geo_feats.shape[1]
        feat_size = child_geo_feats.shape[2]

        # sum over child features (in larger feature space, using hidden_size)
        skip_geo_feats = self.skip_op_geo(child_geo_feats)
        child_geo_feats = F.leaky_relu(self.child_op_geo(child_geo_feats))

        # zero non-existent children
        child_geo_feats = child_geo_feats * child_exists
        skip_geo_feats = skip_geo_feats * child_exists

        child_geo_feats = child_geo_feats.view(batch_size, max_childs, -1)
        skip_geo_feats = skip_geo_feats.view(batch_size, max_childs, -1)

        if self.geo_symmetric_type == 'max':
            parent_feat = F.leaky_relu(child_geo_feats.max(dim=1)[0])
            skip_geo_feat = F.leaky_relu(skip_geo_feats.max(dim=1)[0])
        elif self.geo_symmetric_type == 'sum':
            parent_feat = F.leaky_relu(child_geo_feats.sum(dim=1))
            skip_geo_feat = F.leaky_relu(skip_geo_feats.sum(dim=1))
        elif self.geo_symmetric_type == 'avg':
            parent_feat = F.leaky_relu(child_geo_feats.sum(dim=1) / child_exists.sum(dim=1))
            skip_geo_feat = F.leaky_relu(skip_geo_feats.sum(dim=1) / child_exists.sum(dim=1))
        else:
            raise ValueError(f'Unknown symmetric type: {self.geo_symmetric_type}')

        # back to standard feature space size
        # parent_feat = F.leaky_relu(self.second_geo(parent_feat))
        parent_feat = F.leaky_relu(skip_geo_feat + self.second_norm_geo(self.second_geo(parent_feat)))
        return parent_feat

    def forward(self, child_feats, child_geo_feats, child_exists):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        feat_size = child_feats.shape[2]

        # sum over child features (in larger feature space, using hidden_size)
        skip_feats = self.skip_op(child_feats)
        child_feats = F.leaky_relu(self.child_op(child_feats))

        # zero non-existent children
        child_feats = child_feats * child_exists
        skip_feats = skip_feats * child_exists

        child_feats = child_feats.view(batch_size, max_childs, -1)
        skip_feats = skip_feats.view(batch_size, max_childs, -1)

        if self.struct_symmetric_type == 'max':
            parent_feat = F.leaky_relu(child_feats.max(dim=1)[0])
            skip_feat = F.leaky_relu(skip_feats.max(dim=1)[0])
        elif self.struct_symmetric_type == 'sum':
            parent_feat = F.leaky_relu(child_feats.sum(dim=1))
            skip_feat = F.leaky_relu(skip_feats.sum(dim=1))
        elif self.struct_symmetric_type == 'avg':
            parent_feat = F.leaky_relu(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            skip_feat = F.leaky_relu(skip_feats.sum(dim=1) / child_exists.sum(dim=1))
        else:
            raise ValueError(f'Unknown symmetric type: {self.struct_symmetric_type}')

        # back to standard feature space size
        # parent_feat = F.leaky_relu(self.second(parent_feat))
        parent_feat = F.leaky_relu(skip_feat + self.second_norm(self.second(parent_feat)))
        parent_geo_feat = self.geo_forward(child_geo_feats, child_exists)
        return parent_feat, parent_geo_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config, Tree, MeshInfo, variational=False, probabilistic=True):
        super(RecursiveEncoder, self).__init__()
        self.conf = config
        self.max_part_per_parent = config.max_part_per_parent
        self.semantic_feat_len = config.feature_size//2

        self.box_encoder = BoxEncoder(in_size = 10, feature_size=config.feature_size, hidden_size = config.hidden_size)
        self.surf_encoder = PartDeformEncoder2(num_point = config.num_point, feat_len = config.geo_feat_size, edge_index = MeshInfo.edge_index, probabilistic=False)
        self.mlp_center = nn.Linear(config.geo_feat_size + config.feature_size + 3, config.geo_feat_size)

        self.child_encoder = SymmetricChildEncoder(
                feature_size_struct=config.feature_size, 
                feature_size_geo=config.geo_feat_size, 
                hidden_size=config.hidden_size, 
                symmetric_type=config.node_symmetric_type,
                max_part_per_parent=self.max_part_per_parent,
                Tree = Tree)

        if variational:
            self.sample_struct_encoder = Sampler(feature_size=config.feature_size, \
                    hidden_size=config.hidden_size, probabilistic=probabilistic)
            self.sample_geo_encoder = Sampler(feature_size=config.feature_size, \
                    hidden_size=config.hidden_size, probabilistic=probabilistic)

    def encode_node(self, node):
        if node.is_leaf:
            node_struct_feats = torch.cat([node.geo.mean(dim=1)[0:1], torch.zeros(1, self.conf.feature_size - 3).to(device = node.geo.device)], dim = 1)#torch.zeros(1, self.semantic_feat_len).to(self.conf.device)
            node_geo_feats = self.surf_encoder(node.dggeo)
            node.geo_feat = node_geo_feats
            node_geo_feats = F.leaky_relu(self.mlp_center(torch.cat([node_geo_feats, self.box_encoder(node.get_box_quat().squeeze(1)), node.geo.mean(dim=1)], dim = 1)))
            return node_struct_feats, node_geo_feats
        else:
            # get features of all children
            child_struct_feats = []
            child_geo_feats = []
            for child in node.children:
                node_struct_feats, node_geo_feats = self.encode_node(child)
                cur_child_feat = torch.cat([node_struct_feats, child.get_semantic_one_hot(), child.get_group_ins_one_hot(self.max_part_per_parent)], dim=1)
                child_struct_feats.append(cur_child_feat.unsqueeze(dim=1))
                child_geo_feats.append(node_geo_feats.unsqueeze(dim=1))
            child_struct_feats = torch.cat(child_struct_feats, dim=1)
            child_geo_feats = torch.cat(child_geo_feats, dim=1)

            if child_struct_feats.shape[1] > self.conf.max_child_num:
                raise ValueError('Node has too many children.')

            # pad with zeros
            if child_struct_feats.shape[1] < self.conf.max_child_num:
                padding = child_struct_feats.new_zeros(child_struct_feats.shape[0], \
                        self.conf.max_child_num-child_struct_feats.shape[1], child_struct_feats.shape[2])
                child_struct_feats = torch.cat([child_struct_feats, padding], dim=1)
            if child_geo_feats.shape[1] < self.conf.max_child_num:
                padding = child_geo_feats.new_zeros(child_geo_feats.shape[0], \
                        self.conf.max_child_num-child_geo_feats.shape[1], child_geo_feats.shape[2])
                child_geo_feats = torch.cat([child_geo_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_struct_feats.new_zeros(child_struct_feats.shape[0], self.conf.max_child_num, 1)
            child_exists[:, :len(node.children), :] = 1

            # get feature of current node (parent of the children)
            return self.child_encoder(child_struct_feats, child_geo_feats, child_exists)

    def encode_structure(self, obj):
        root_latent_struct, root_latent_geo = self.encode_node(obj)
        root_latent_struct = self.sample_struct_encoder(root_latent_struct).repeat(root_latent_geo.size(0), 1)
        root_latent_geo = self.sample_geo_encoder(root_latent_geo)
        root_latent = torch.cat([root_latent_struct[:,:self.conf.feature_size], root_latent_geo[:,:self.conf.feature_size], root_latent_struct[:,self.conf.feature_size:], root_latent_geo[:,self.conf.feature_size:]], dim = 1)
        # root_latent = self.sample_encoder(root_latent_struct, root_latent_geo)
        return root_latent


class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)
        self.skip = nn.Linear(feature_size, 1)

    def forward(self, input_feature):
        output = F.leaky_relu(self.mlp1(input_feature))
        output = self.skip(input_feature) + self.mlp2(output)
        return output

class LeafCenterer(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafCenterer, self).__init__()
        self.skip = nn.Linear(feature_size, 3)
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 3)

    def forward(self, input_feature):
        output = F.leaky_relu(self.mlp1(input_feature))
        output = torch.tanh(self.skip(input_feature) + self.mlp2(output))
        return output

class LeafGeo(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafGeo, self).__init__()
        self.skip = nn.Linear(feature_size, feature_size)
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, feature_size)

    def forward(self, input_feature):
        output = F.leaky_relu(self.mlp1(input_feature))
        output = F.leaky_relu(self.mlp2(output))
        output = self.skip(input_feature) + self.mlp3(output)
        return output

class SampleDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.skip = nn.Linear(feature_size, feature_size)

    def forward(self, input_feat):
        feat2 = F.leaky_relu(self.mlp1(input_feat))
        output = F.leaky_relu(self.skip(input_feat) + self.mlp2(feat2))
        return output


class BoxDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(BoxDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.center = nn.Linear(hidden_size, 3)
        self.size = nn.Linear(hidden_size, 3)
        self.quat = nn.Linear(hidden_size, 4)
        self.center_skip = nn.Linear(feature_size, 3)
        self.size_skip = nn.Linear(feature_size, 3)
        self.quat_skip = nn.Linear(feature_size, 4)

    def forward(self, parent_feature):
        feat = F.leaky_relu(self.mlp(parent_feature))
        center = torch.tanh(self.center_skip(parent_feature)+self.center(feat))
        size = torch.sigmoid(self.size_skip(parent_feature)+self.size(feat)) * 2
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = (self.quat_skip(parent_feature)+self.quat(feat)).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        vector = torch.cat([center, size, quat], dim=1)
        return vector


class ConcatChildDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num, max_part_per_parent, Tree):
        super(ConcatChildDecoder, self).__init__()

        self.max_child_num = max_child_num
        self.max_part_per_parent = max_part_per_parent
        self.hidden_size = hidden_size
        self.Tree = Tree

        self.mlp_parent = nn.Linear(feature_size, hidden_size*max_child_num)
        self.mlp_exists = nn.Linear(hidden_size, 1)
        self.mlp_sem = nn.Linear(hidden_size, Tree.num_sem)
        self.mlp_sem_ins = nn.Linear(hidden_size, self.max_part_per_parent)
        self.mlp_child = nn.Linear(hidden_size, feature_size)
        self.norm_child = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)

        self.mlp_geo_child = nn.Linear(hidden_size, feature_size)
        self.mlp_geo_parent = nn.Linear(feature_size, hidden_size*max_child_num)
        self.norm_geo_child = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)

    def geo_forward(self, parent_geo_feature):
        batch_size = parent_geo_feature.shape[0]
        feat_size = parent_geo_feature.shape[1]

        parent_geo_feature = F.leaky_relu(self.mlp_geo_parent(parent_geo_feature))

        # node features
        child_geo_feats = self.norm_geo_child(self.mlp_geo_child(parent_geo_feature.view(-1, self.hidden_size)))
        child_geo_feats = child_geo_feats.view(batch_size, self.max_child_num, feat_size)
        child_geo_feats = F.leaky_relu(child_geo_feats)

        return child_geo_feats

    def forward(self, parent_struct_feature, parent_geo_feature):
        batch_size = parent_struct_feature.shape[0]
        feat_size = parent_struct_feature.shape[1]

        parent_struct_feature = F.leaky_relu(self.mlp_parent(parent_struct_feature))
        child_feats = parent_struct_feature.view(batch_size, self.max_child_num, self.hidden_size)

        child_geo_feature = self.geo_forward(parent_geo_feature)

        # node existence
        child_exists_logits = self.mlp_exists(child_feats.view(-1, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, self.max_child_num, 1)

        # node semantics
        child_sem_logits = self.mlp_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, self.max_child_num, self.Tree.num_sem)

        # node ins semantics
        child_sem_ins_logits = self.mlp_sem_ins(child_feats.view(-1, self.hidden_size))
        child_sem_ins_logits = child_sem_ins_logits.view(batch_size, self.max_child_num, self.max_part_per_parent)

        # node features
        child_feats = self.norm_child(self.mlp_child(parent_struct_feature.view(-1, self.hidden_size)))
        child_feats = child_feats.view(batch_size, self.max_child_num, feat_size)
        child_feats = F.leaky_relu(child_feats)

        return child_feats, child_geo_feature, child_sem_logits, child_sem_ins_logits, child_exists_logits


class RecursiveDecoder(nn.Module):

    def __init__(self, config, Tree, MeshInfo, no_center = False):
        super(RecursiveDecoder, self).__init__()

        self.conf = config
        self.Tree = Tree
        self.no_center = no_center
        self.max_part_per_parent = config.max_part_per_parent

        if MeshInfo is not None:
            self.dggeo2vertex = Feature2Vertex_pytorch(MeshInfo, config.gpu)
        else:
            self.dggeo2vertex = None

        self.box_decoder = BoxDecoder(config.geo_feat_size, config.hidden_size)
        self.surf_decoder = PartDeformDecoder2(feat_len = config.geo_feat_size, num_point = config.num_point, edge_index = MeshInfo.edge_index, structure_feat_len = 3)
        self.geo_mlp = LeafGeo(config.feature_size, config.geo_feat_size)
        self.leaf_center = LeafCenterer(config.feature_size, config.hidden_size)

        self.child_decoder = ConcatChildDecoder(
                feature_size=config.feature_size, 
                hidden_size=config.hidden_size, 
                max_child_num=config.max_child_num,
                max_part_per_parent = config.max_part_per_parent,
                Tree = Tree)

        # self.sample_decoder = SampleDecoder(config.feature_size, config.hidden_size)
        self.sample_struct_decoder = SampleDecoder(config.feature_size, config.hidden_size)
        self.sample_geo_decoder = SampleDecoder(config.feature_size, config.hidden_size)

        self.leaf_classifier = LeafClassifier(config.feature_size, config.hidden_size)

        self.bceLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.chamferLoss = ChamferDistance()
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.mseLoss = nn.MSELoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))

    def boxLossEstimator(self, box_feature, gt_box_feature):
        if self.no_center:
            box_feature[:, :3] = gt_box_feature[:, :3]
        pred_box_pc = transform_pc_batch(self.unit_cube, box_feature)
        with torch.no_grad():
            pred_reweight = get_surface_reweighting_batch(box_feature[:, 3:6], self.unit_cube.size(0))
        gt_box_pc = transform_pc_batch(self.unit_cube, gt_box_feature)
        with torch.no_grad():
            gt_reweight = get_surface_reweighting_batch(gt_box_feature[:, 3:6], self.unit_cube.size(0))
        dist1, dist2 = self.chamferLoss(gt_box_pc, pred_box_pc)
        loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
        loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
        loss = (loss1 + loss2) / 2
        return loss

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    def acap2coor(self, deformed_coor, center):
        num_point = deformed_coor.size(1)
        geo_local = self.dggeo2vertex.get_vertex_from_feature(deformed_coor)
        geo_local = geo_local - geo_local.mean(dim=1).unsqueeze(dim=1).repeat(1, num_point, 1) + center.unsqueeze(dim=1).repeat(1, num_point, 1)
        return geo_local

    def box2surf(self, box_feature, geo_feature, full_label):

        box_feature_aftermlp = self.geo_mlp(geo_feature)
        center = self.leaf_center(geo_feature)
        pred_acap = self.surf_decoder(torch.cat([box_feature_aftermlp, center],dim =1))

        if self.dggeo2vertex is not None:
            deformed_coor = self.acap2coor(pred_acap, center)
        return deformed_coor, pred_acap

    def surfLossEstimator(self, box_feature, geo_feature, gt_node):
        # num_point = self.surf_cube.size(0)
        geo_latent = gt_node.geo_feat
        box_feature_aftermlp = self.geo_mlp(geo_feature)
        center = self.leaf_center(geo_feature)
        recon_acap = self.surf_decoder(torch.cat([geo_latent, gt_node.geo.mean(dim=1)],dim =1))
        pred_acap = self.surf_decoder(torch.cat([box_feature_aftermlp, center],dim =1))

        loss_mapping = self.mseLoss(geo_latent, box_feature_aftermlp).mean()*50
        # loss_center = (self.mseLoss(center_in_struct, gt_node.geo.mean(dim=1)[0:1]).mean())*10
        loss_center = (self.mseLoss(center, gt_node.geo.mean(dim=1)).mean())*15
        loss_acap = self.mseLoss(gt_node.dggeo, recon_acap).mean()*50

        pred_coor = self.acap2coor(pred_acap, center)
        loss_surf = self.mseLoss(pred_coor, gt_node.geo).mean()*40
        # loss_surf = earth_mover_distance(pred_coor, gt_node.geo, transpose=False)/self.surf_cube.size(0)*10.0
        # loss_surf = torch.zeros_like(loss_mapping)
        loss = loss_mapping + loss_surf + loss_acap + loss_center

        # print(loss_mapping)
        # print(loss_surf)
        # print(loss_acap)
        # print(loss_center)
        return loss

    # decode a root code into a tree structure
    def decode_structure(self, z, max_depth):
        root_strcut_latent, root_geo_latent = torch.chunk(z, 2, 1)
        root_strcut_latent = self.sample_struct_decoder(root_strcut_latent)
        root_geo_latent = self.sample_geo_decoder(root_geo_latent)
        root = self.decode_node(root_strcut_latent, root_geo_latent, max_depth, self.Tree.root_sem)
        obj = self.Tree(root=root)
        return obj

    # decode a part node
    def decode_node(self, node_struct_latent, node_geo_latent, max_depth, full_label, is_leaf=False):
        if node_struct_latent.shape[0] != 1:
            raise ValueError('Node decoding does not support batch_size > 1.')

        is_leaf_logit = self.leaf_classifier(node_struct_latent)
        node_is_leaf = is_leaf_logit.item() > 0

        # use maximum depth to avoid potential infinite recursion
        if max_depth < 1:
            is_leaf = True

        # decode the current part box
        # box = self.box_decoder(node_struct_latent)

        if node_is_leaf or is_leaf:
            local_geo, pred_dggeo = self.box2surf(node_struct_latent, node_geo_latent, full_label)
            ret = self.Tree.Node(is_leaf=True, \
                    full_label=full_label, label=full_label.split('/')[-1], geo = local_geo, dggeo = pred_dggeo)
            # ret.set_from_box_quat(box.view(-1))
            return ret
        else:
            child_struct_feats, child_geo_feats, child_sem_logits, child_sem_ins_logits, child_exists_logit = \
                    self.child_decoder(node_struct_latent, node_geo_latent)

            child_sem_logits = child_sem_logits.cpu().detach().numpy().squeeze()
            child_sem_ins_logits = child_sem_ins_logits.cpu().detach().numpy().squeeze()

            # children
            child_nodes = []
            for ci in range(child_struct_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits[ci, self.Tree.part_name2cids[full_label]])
                    idx = self.Tree.part_name2cids[full_label][idx]
                    child_full_label = self.Tree.part_id2name[idx]
                    child_nodes.append(self.decode_node(\
                            child_struct_feats[:, ci, :], child_geo_feats[:, ci, :], max_depth-1, child_full_label, \
                            is_leaf=(child_full_label not in self.Tree.part_non_leaf_sem_names)))
            if len(child_nodes)==0:
                ret = self.Tree.Node(is_leaf=True, \
                    full_label=full_label, label=full_label.split('/')[-1])
            else:
                ret = self.Tree.Node(is_leaf=False, children=child_nodes, \
                    full_label=full_label, label=full_label.split('/')[-1])

            # ret.set_from_box_quat(box.view(-1))
            return ret

    # use gt structure, compute the reconstruction losses
    def structure_recon_loss(self, z, gt_tree):
        root_strcut_latent, root_geo_latent = torch.chunk(z, 2, 1)
        root_strcut_latent = self.sample_struct_decoder(root_strcut_latent[0].unsqueeze(dim=0))
        root_geo_latent = self.sample_geo_decoder(root_geo_latent)
        losses = self.node_recon_loss(root_strcut_latent, root_geo_latent, gt_tree)
        return losses

    # use gt structure, compute the reconstruction losses
    def node_recon_loss(self, root_strcut_latent, root_geo_latent, gt_node):
        if gt_node.is_leaf:
            box = self.box_decoder(root_geo_latent)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().squeeze(1)).mean()
            surf_loss = self.surfLossEstimator(root_strcut_latent, root_geo_latent, gt_node)#torch.zeros_like(box_loss)
            is_leaf_logit = self.leaf_classifier(root_strcut_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))
            return {'box': box_loss+surf_loss, 'leaf': is_leaf_loss, 'anchor': torch.zeros_like(surf_loss), 
                    'exists': torch.zeros_like(surf_loss), 'semantic': torch.zeros_like(surf_loss), 
                    'edge_exists': torch.zeros_like(surf_loss), 
                    'sym': torch.zeros_like(surf_loss), 'adj': torch.zeros_like(surf_loss)}
        else:
            child_struct_feats, child_geo_feats, child_sem_logits, child_sem_ins_logits, child_exists_logits = \
                    self.child_decoder(root_strcut_latent, root_geo_latent)

            # generate box prediction for each child
            feature_len = root_strcut_latent.size(1)
            # child_pred_boxes = self.box_decoder(torch.cat([child_struct_feats.repeat(child_geo_feats.size(0), 1, 1).view(-1, feature_len), \
            #                                                child_geo_feats.view(-1, feature_len)], dim = 1)).view(child_geo_feats.size(0), -1, 10)
            child_pred_boxes = self.box_decoder(child_geo_feats.view(-1, feature_len)).view(child_geo_feats.size(0), -1, 10)
            num_child_parts = child_pred_boxes.size(1)

            # perform hungarian matching between pred boxes and gt boxes
            with torch.no_grad():
                child_gt_boxes = torch.cat([child_node.get_box_quat()[0] for child_node in gt_node.children], dim=0)
                num_gt = child_gt_boxes.size(0)

                child_pred_boxes_tiled = child_pred_boxes[0].unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_boxes_tiled = child_gt_boxes.unsqueeze(dim=1).repeat(1, num_child_parts, 1)

                dist_mat = self.boxLossEstimator(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_child_parts)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

            # train the current node to be non-leaf
            is_leaf_logit = self.leaf_classifier(root_strcut_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1))

            # # train the current node box to gt
            # box = self.box_decoder(torch.cat([root_strcut_latent.repeat(root_geo_latent.size(0), 1), root_geo_latent], dim = 1))
            box = self.box_decoder(root_geo_latent)
            box_loss = self.boxLossEstimator(box, gt_node.get_box_quat().squeeze(1)).mean()
            # print(box_loss)

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_sem_ins_gt_labels = []
            child_sem_ins_pred_logits = []
            # child_box_gt = []
            # child_box_pred = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id())
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_sem_ins_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_group_ins_id())
                child_sem_ins_pred_logits.append(child_sem_ins_logits[0, matched_pred_idx[i], :].view(1, -1))
                # child_box_gt.append(gt_node.children[matched_gt_idx[i]].get_box_quat())
                # child_box_pred.append(child_pred_boxes[matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1
                
            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, \
                    device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)

            child_sem_ins_pred_logits = torch.cat(child_sem_ins_pred_logits, dim=0)
            child_sem_ins_gt_labels = torch.tensor(child_sem_ins_gt_labels, dtype=torch.int64, \
                    device=child_sem_ins_pred_logits.device)
            semantic_ins_loss = self.semCELoss(child_sem_ins_pred_logits, child_sem_ins_gt_labels)
            semantic_loss = semantic_loss.sum() + semantic_ins_loss.sum()

            # train unused boxes to zeros
            unmatched_boxes = []
            for i in range(num_child_parts):
                if i not in matched_pred_idx:
                    unmatched_boxes.append(child_pred_boxes[:, i, 3:6])
            if len(unmatched_boxes) > 0:
                unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
                unused_box_loss = unmatched_boxes.pow(2).sum() * 0.01
            else:
                unused_box_loss = 0.0

            # train exist scores
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits, target=child_exists_gt, reduction='none')
            child_exists_loss = child_exists_loss.sum()

            # calculate children + aggregate losses
            for i in range(len(matched_gt_idx)):
                child_losses = self.node_recon_loss(\
                        child_struct_feats[:, matched_pred_idx[i], :], child_geo_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]])
                box_loss = box_loss + child_losses['box']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']

            return {'box': box_loss + unused_box_loss, 'leaf': is_leaf_loss, 'anchor': torch.zeros_like(box_loss), 
                    'exists': child_exists_loss, 'semantic': semantic_loss,
                    'edge_exists': torch.zeros_like(box_loss), 
                    'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}


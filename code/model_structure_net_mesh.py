
"""
    This file defines the box-represented shape VAE/AE model (no edge) for scene, until to part level box geometry.
    borrow from structurenet only box
    the leaf node can predict the box parameter, with skip connenction operation
    log:
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from chamfer_distance import ChamferDistance
# from PyTorchEMD.emd import earth_mover_distance
# from data import Tree
import time, random
from utils import linear_assignment, load_pts, transform_pc_batch, get_surface_reweighting_batch, qrot

from feature2vertex import Feature2Vertex_pytorch
from model_part_deform import PartDeformEncoder2, PartDeformDecoder2

class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = torch.nn.functional.leaky_relu(self.mlp1(x), 0.1)
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu

class BoxEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size, orient = False):
        super(BoxEncoder, self).__init__()

        self.mlp_skip = nn.Linear(10, feature_size)
        self.mlp1 = nn.Sequential(nn.Linear(10, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, hidden_size))
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, box_input):
        net = F.leaky_relu(self.mlp1(box_input), 0.1)
        net = F.leaky_relu(self.mlp_skip(box_input) + self.mlp2(net), 0.1)
        # box_vector = torch.nn.functional.leaky_relu(self.encoder(box_input), 0.1)
        return net

class SymmetricChildEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size, symmetric_type, Tree):
        super(SymmetricChildEncoder, self).__init__()

        print(f'Using Symmetric Type: {symmetric_type}')
        self.symmetric_type = symmetric_type

        self.child_op_part = nn.Linear(feature_size + Tree.part_num_sem, hidden_size)
        self.second_part = nn.Linear(hidden_size, feature_size)

        # skip connection
        self.second_norm = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)
        self.skip_op_part = nn.Linear(feature_size + Tree.part_num_sem, feature_size)

    def forward(self, child_feats, child_exists):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        feat_size = child_feats.shape[2]
        # print(child_feats.shape)
        max_childs = 10

        # sum over child features (in larger feature space, using hidden_size)
        skip_feats = F.leaky_relu(self.skip_op_part(child_feats), 0.1)
        child_feats = F.leaky_relu(self.child_op_part(child_feats), 0.1)

        # print(child_feats.shape)
        # print(child_exists.shape)

        # zero non-existent children
        child_feats = child_feats * child_exists
        skip_feats = skip_feats * child_exists

        child_feats = child_feats.view(batch_size, max_childs, -1)
        skip_feats = skip_feats.view(batch_size, max_childs, -1)
        # print(child_feats.shape)

        if self.symmetric_type == 'max':
            parent_feat = child_feats.max(dim=1)[0]
            skip_feat = skip_feats.max(dim=1)[0]
        elif self.symmetric_type == 'sum':
            parent_feat = child_feats.sum(dim=1)
            skip_feat = skip_feats.sum(dim=1)
        elif self.symmetric_type == 'avg':
            parent_feat = child_feats.sum(dim=1) / child_exists.sum(dim=1)
            skip_feat = skip_feats.sum(dim=1) / child_exists.sum(dim=1)
        else:
            raise ValueError(f'Unknown symmetric type: {self.symmetric_type}')

        # back to standard feature space size
        # parent_feat = torch.nn.functional.leaky_relu(second(parent_feat), 0.1)
        parent_feat = F.leaky_relu(skip_feat + self.second_norm(self.second_part(parent_feat)), 0.1)
        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, config, Tree, meshinfo, variational=False, probabilistic=True, orient = False):
        super(RecursiveEncoder, self).__init__()
        self.conf = config
        self.orient = orient
        self.Tree = Tree

        self.box_encoder = BoxEncoder(feature_size=config.feature_size, hidden_size = config.hidden_size, orient = orient)
        self.surf_encoder = PartDeformEncoder2(num_point = meshinfo.point_num, feat_len = config.geopart_feat_size, edge_index = meshinfo.edge_index, probabilistic=False)
        self.mlp_center = nn.Linear(config.geopart_feat_size + config.feature_size + 3, config.geopart_feat_size)

        self.child_encoder = SymmetricChildEncoder(
                feature_size=config.feature_size,
                hidden_size=config.hidden_size,
                symmetric_type=config.node_symmetric_type,
                Tree = Tree)

        if variational:
            self.sample_encoder = Sampler(feature_size=config.feature_size, \
                    hidden_size=config.hidden_size, probabilistic=probabilistic)

    def encode_node(self, node):
        max_child_num = 10

        if node.is_leaf:
            # return self.box_encoder(node.get_box_quat())
            if self.orient:
                box_feature = torch.cat((node.get_box_quat(), node.orient), dim = 1)
            else:
                box_feature = node.get_box_quat()
            node_geo_feats = self.surf_encoder(node.dggeo)
            node.geo_feat = node_geo_feats
            all_feature = F.leaky_relu(self.mlp_center(torch.cat([node_geo_feats, self.box_encoder(box_feature), node.geo.mean(dim=1)], dim = 1)))

            return all_feature
        else:
            # get features of all children
            child_feats = []
            for child in node.children:
                # print(child.full_label)
                cur_child_feat = torch.cat([self.encode_node(child), child.get_semantic_one_hot(self.Tree)], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            if child_feats.shape[1] > self.conf.max_child_num:
                raise ValueError('Node has too many children.')

            # # pad with zeros
            if child_feats.shape[1] < max_child_num:
                padding = child_feats.new_zeros(child_feats.shape[0], \
                        max_child_num-child_feats.shape[1], child_feats.shape[2])
                child_feats = torch.cat([child_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_feats.new_zeros(child_feats.shape[0], child_feats.shape[1], 1)
            child_exists[:, :len(node.children), :] = 1

            # get feature of current node (parent of the children)
            return self.child_encoder(child_feats, child_exists)

    def encode_structure(self, obj):
        root_latent = self.encode_node(obj.root)
        return self.sample_encoder(root_latent)


class LeafClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafClassifier, self).__init__()
        self.skip = nn.Linear(feature_size, 1)
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)

    def forward(self, input_feature):
        output = torch.nn.functional.leaky_relu(self.mlp1(input_feature), 0.1)
        output = self.mlp2(output) + self.skip(input_feature)
        return output

class SampleDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.skip = nn.Linear(feature_size,feature_size)

    def forward(self, input_feature):
        output = torch.nn.functional.leaky_relu(self.mlp1(input_feature), 0.1)
        output = F.leaky_relu(self.skip(input_feature) + self.mlp2(output), 0.1)
        return output

class BoxDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, orient):
        super(BoxDecoder, self).__init__()
        self.orient = orient
        self.mlp = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, hidden_size))
        self.center = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, 3))
        self.size = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, 3))
        self.quat = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1), nn.Linear(hidden_size, 4))
        self.center_skip = nn.Linear(feature_size, 3)
        self.size_skip = nn.Linear(feature_size, 3)
        self.quat_skip = nn.Linear(feature_size, 4)
        if orient:
            self.orient = nn.Linear(hidden_size, 4)

    def forward(self, parent_feature):
        feat = torch.nn.functional.leaky_relu(self.mlp(parent_feature), 0.1)
        center = torch.tanh(self.center(feat) + self.center_skip(parent_feature))
        size = torch.sigmoid(self.size_skip(parent_feature)+self.size(feat)) * 2
        quat_bias = feat.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        quat = (self.quat_skip(parent_feature) + self.quat(feat)).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=1).unsqueeze(dim=1).sqrt())
        if self.orient:
            oriten = self.orient(feat).add(quat_bias)
            vector = torch.cat([center, size, quat, oriten], dim=1)
        else:
            vector = torch.cat([center, size, quat], dim=1)
        return vector

class LeafCenterer(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafCenterer, self).__init__()
        self.skip = nn.Linear(feature_size, 3)
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 3)

    def forward(self, input_feature):
        output = F.leaky_relu(self.mlp1(input_feature))
        output = torch.tanh(self.skip(input_feature) + self.mlp2(output)) *2
        return output

class LeafGeo(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(LeafGeo, self).__init__()
        self.skip = nn.Linear(feature_size, feature_size)
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp4 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, feature_size)

    def forward(self, input_feature):
        output = F.leaky_relu(self.mlp1(input_feature),0.1)
        output = F.leaky_relu(self.mlp2(output),0.1)
        output = F.leaky_relu(self.mlp4(output),0.1)
        output = self.skip(input_feature) + self.mlp3(output)
        return output

class ConcatChildDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size, max_child_num, Tree):
        super(ConcatChildDecoder, self).__init__()

        self.max_part_num = 10
        self.hidden_size = hidden_size
        self.Tree = Tree

        self.mlp_parent_part = nn.Linear(feature_size, hidden_size*self.max_part_num)
        self.mlp_exists_part = nn.Linear(hidden_size, 1)
        self.mlp_part_sem = nn.Linear(hidden_size, Tree.part_num_sem)
        self.mlp_child_part = nn.Linear(hidden_size, feature_size)
        self.norm_child_part = nn.GroupNorm(num_groups=min(32, feature_size//8), num_channels=feature_size)

    def forward(self, parent_feature):
        batch_size = parent_feature.shape[0]
        feat_size = parent_feature.shape[1]

        max_child_num = self.max_part_num
        num_sem = self.Tree.part_num_sem

        parent_feature = F.leaky_relu(self.mlp_parent_part(parent_feature), 0.1)
        child_feats = parent_feature.view(batch_size, max_child_num, self.hidden_size)

        # node existence
        child_exists_logits = self.mlp_exists_part(child_feats.view(-1, self.hidden_size))
        child_exists_logits = child_exists_logits.view(batch_size, max_child_num, 1)

        # node room semantics
        child_sem_logits = self.mlp_part_sem(child_feats.view(-1, self.hidden_size))
        child_sem_logits = child_sem_logits.view(batch_size, max_child_num, num_sem)

        # node features
        child_feats = self.norm_child_part(self.mlp_child_part(parent_feature.view(-1, self.hidden_size)))
        child_feats = child_feats.view(batch_size, max_child_num, feat_size)
        child_feats = F.leaky_relu(child_feats, 0.1)

        return child_feats, child_sem_logits, child_exists_logits

class RecursiveDecoder(nn.Module):

    def __init__(self, config, Tree, meshinfo, orient = False, no_center = False):
        super(RecursiveDecoder, self).__init__()

        self.conf = config
        self.Tree = Tree
        self.no_center = no_center
        self.orient = orient

        if meshinfo is not None:
            self.dggeo2vertex = Feature2Vertex_pytorch(meshinfo, config.gpu)
        else:
            self.dggeo2vertex = None

        self.box_decoder = BoxDecoder(config.feature_size, config.hidden_size, orient = self.orient)
        self.surf_decoder = PartDeformDecoder2(feat_len = config.geopart_feat_size, num_point = meshinfo.point_num, edge_index = meshinfo.edge_index, structure_feat_len = 3)
        self.geo_mlp = LeafGeo(config.feature_size, config.geopart_feat_size)
        self.leaf_center = LeafCenterer(config.feature_size, config.hidden_size)

        self.child_decoder = ConcatChildDecoder(
                feature_size=config.feature_size,
                hidden_size=config.hidden_size,
                max_child_num=config.max_child_num,
                Tree = Tree)

        self.sample_decoder = SampleDecoder(config.feature_size, config.hidden_size)
        self.leaf_classifier = LeafClassifier(config.feature_size, config.hidden_size)

        self.bceLoss = nn.BCEWithLogitsLoss(reduction='none')
        self.chamferLoss = ChamferDistance()
        self.semCELoss = nn.CrossEntropyLoss(reduction='none')
        self.mseLoss = nn.SmoothL1Loss(reduction='none')
        # self.L1Loss = nn.L1Loss(reduction='none')

        self.register_buffer('unit_cube', torch.from_numpy(load_pts('cube.pts')))

    def boxLossEstimator1(self, box_feature, gt_box_feature):

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

    def boxLossEstimator(self, box_feature, gt_box_feature):

        pred_box_pc = transform_pc_batch(self.unit_cube, box_feature[:,:10])
        gt_box_pc = transform_pc_batch(self.unit_cube, gt_box_feature[:,:10])
        dist1, dist2 = self.chamferLoss(gt_box_pc, pred_box_pc)
        loss1 = dist1.sum(dim=1) / dist1.size(1)
        loss2 = dist2.sum(dim=1) / dist2.size(1)
        loss = (loss1 + loss2) / 2

        return loss

    def acap2coor(self, deformed_coor, center):
        num_point = deformed_coor.size(1)
        geo_local = self.dggeo2vertex.get_vertex_from_feature(deformed_coor)
        geo_local = geo_local - geo_local.mean(dim=1).unsqueeze(dim=1).repeat(1, num_point, 1) + center.unsqueeze(dim=1).repeat(1, num_point, 1)
        return geo_local

    def box2surf(self, geo_feature, full_label):
        box_feature_aftermlp = self.geo_mlp(geo_feature)
        center = self.leaf_center(geo_feature)
        pred_acap = self.surf_decoder(torch.cat([box_feature_aftermlp, center],dim =1))

        if self.dggeo2vertex is not None:
            deformed_coor = self.acap2coor(pred_acap, center)
            # import trimesh
            # obb_mesh = trimesh.Trimesh(vertices = deformed_coor[0].detach().cpu().numpy())
            # obb_mesh.export(str(time.time()).replace('.', '_')+'_'+str(random.random()).replace('.', '_') + 'temp3_500.obj')
        return deformed_coor, pred_acap

    def surfLossEstimator(self, geo_feature, gt_node):

        geo_latent = gt_node.geo_feat
        box_feature_aftermlp = self.geo_mlp(geo_feature)
        center = self.leaf_center(geo_feature)
        recon_acap = self.surf_decoder(torch.cat([geo_latent, gt_node.geo.mean(dim=1)],dim =1))
        pred_acap = self.surf_decoder(torch.cat([box_feature_aftermlp, center],dim =1))

        loss_mapping = self.mseLoss(geo_latent, box_feature_aftermlp).mean()*500
        loss_center = (self.mseLoss(center, gt_node.geo.mean(dim=1)).mean())*500
        loss_acap = self.mseLoss(gt_node.dggeo, recon_acap).mean()*0.01

        pred_coor = self.acap2coor(pred_acap, center)
        loss_surf = self.mseLoss(pred_coor, gt_node.geo).mean()*10

        return loss_surf, loss_acap, loss_center, loss_mapping

    def orientLoss(self, orient, gt_orient):
        # print(orient.shape)
        # print(gt_orient.shape)
        pred_orient = qrot(orient.view(-1, 4), torch.tensor(np.array([0.0,0.0,1.0]), dtype=torch.float32, device=orient.device).view(1, 3)).view(1, 3)
        gt_orient = qrot(gt_orient.view(-1, 4), torch.tensor(np.array([0.0,0.0,1.0]), dtype=torch.float32, device=orient.device).view(1, 3)).view(1, 3)
        loss = self.L1Loss(pred_orient, gt_orient).mean()

        return loss

    def geoToGlobal(self, geo_local, geo_center, geo_scale):
        num_point = geo_local.size(1)
        return (geo_local - geo_local.mean(dim=1).unsqueeze(dim=1).repeat(1, num_point, 1)) * geo_scale.unsqueeze(dim=1).repeat(1, num_point, 3) + \
                geo_center.unsqueeze(dim=1).repeat(1, num_point, 1)

    def chamferDist(self, pc1, pc2):
        dist1, dist2 = self.chamferLoss(pc1, pc2)
        return (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2

    def isLeafLossEstimator(self, is_leaf_logit, gt_is_leaf):
        return self.bceLoss(is_leaf_logit, gt_is_leaf).view(-1)

    # decode a root code into a tree structure
    def decode_structure(self, z, max_depth):
        root_latent = self.sample_decoder(z)
        root = self.decode_node(root_latent, max_depth, self.Tree.root_sem)
        obj = self.Tree(root=root)
        return obj

    # decode a part node
    def decode_node(self, node_latent, max_depth, full_label, is_leaf=False):
        if node_latent.shape[0] != 1:
            raise ValueError('Node decoding does not support batch_size > 1.')

        is_leaf_logit = self.leaf_classifier(node_latent)
        node_is_leaf = is_leaf_logit.item() > 0

        # use maximum depth to avoid potential infinite recursion
        if max_depth < 1:
            is_leaf = True

        if node_is_leaf or is_leaf:
            local_geo, pred_dggeo = self.box2surf(node_latent, full_label)
            ret = self.Tree.Node(is_leaf=True, \
                    full_label = full_label, label=full_label.split('/')[-1], geo = local_geo, dggeo = pred_dggeo)

            if self.orient:
                ret.orient = qrot(box.view(-1)[10:].view(-1, 4), torch.tensor(np.array([0.0,0.0,1.0]), dtype=torch.float32, device=box.device).view(1, 3)).view(1, 3)
            return ret
        else:
            child_feats, child_sem_logits, child_exists_logit = \
                    self.child_decoder(node_latent)

            child_sem_logits = child_sem_logits.cpu().detach().numpy().squeeze()
            # print(child_sem_logits.shape)

            # children
            child_nodes = []
            for ci in range(child_feats.shape[1]):
                if torch.sigmoid(child_exists_logit[:, ci, :]).item() > 0.5:
                    idx = np.argmax(child_sem_logits[ci, self.Tree.part_name2cids[full_label]])
                    idx = self.Tree.part_name2cids[full_label][idx]
                    child_full_label = self.Tree.part_id2name[idx]
                    child_nodes.append(self.decode_node(\
                            child_feats[:, ci, :], max_depth-1, child_full_label, \
                            is_leaf=(child_full_label not in self.Tree.part_non_leaf_sem_names)))
            if len(child_nodes)==0:
                ret = self.Tree.Node(is_leaf=True, full_label = full_label, label=full_label.split('/')[-1])
            else:
                ret = self.Tree.Node(is_leaf=False, children=child_nodes, full_label = full_label, label=full_label.split('/')[-1])

            # ret.set_from_box_quat(box.view(-1)[:10])
            return ret

    # use gt structure, compute the reconstruction losses
    def structure_recon_loss(self, z, gt_tree):
        root_latent = self.sample_decoder(z)
        losses = self.node_recon_loss(root_latent, gt_tree.root)
        return losses

    # use gt structure, compute the reconstruction losses
    def node_recon_loss(self, node_latent, gt_node):

        # geo_local, geo_center, geo_scale, geo_feat, box_quat = self.node_decoder(node_latent)
        # geo_global = self.geoToGlobal(geo_local, geo_center, geo_scale)

        if gt_node.is_leaf:

            box = self.box_decoder(node_latent)
            box_loss = self.boxLossEstimator1(box, gt_node.get_box_quat().view(1, -1))
            loss_surf, loss_acap, loss_center, loss_mapping = self.surfLossEstimator(node_latent, gt_node)#torch.zeros_like(box_loss)
            # latent_loss = torch.zeros_like(box_loss)
            # center_loss = torch.zeros_like(box_loss)
            scale_loss = torch.zeros_like(box_loss)
            # geo_loss = torch.zeros_like(box_loss)
            if self.orient:
                orient_loss = self.orientLoss(box[:, 10:], gt_node.orient.view(1, -1))
            else:
                orient_loss = torch.zeros_like(box_loss)
            is_leaf_logit = self.leaf_classifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1)).mean()

            return {'box': box_loss + orient_loss, 'surf': loss_surf, 'center':loss_center + scale_loss, 'feat': loss_mapping, 'leaf': is_leaf_loss,
                    'acap': loss_acap,
                    'anchor': torch.zeros_like(box_loss), 'exists': torch.zeros_like(box_loss), 'semantic': torch.zeros_like(box_loss),
                    'edge_exists': torch.zeros_like(box_loss),
                    'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}
        else:
            child_feats, child_sem_logits, child_exists_logits = \
                    self.child_decoder(node_latent)

            # generate box prediction for each child
            feature_len = node_latent.size(1)
            child_pred_boxes = self.box_decoder(child_feats.view(-1, feature_len))
            num_child_parts = child_pred_boxes.size(0)

            # perform hungarian matching between pred boxes and gt boxes
            with torch.no_grad():
                child_gt_boxes = torch.cat([child_node.get_box_quat() for child_node in gt_node.children], dim=0)
                num_gt = child_gt_boxes.size(0)

                child_pred_boxes_tiled = child_pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                child_gt_boxes_tiled = child_gt_boxes.unsqueeze(dim=1).repeat(1, num_child_parts, 1)

                dist_mat = self.boxLossEstimator1(child_gt_boxes_tiled.view(-1, 10), child_pred_boxes_tiled.view(-1, 10)).view(-1, num_gt, num_child_parts)

                _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat)

            # train the current node to be non-leaf
            is_leaf_logit = self.leaf_classifier(node_latent)
            is_leaf_loss = self.isLeafLossEstimator(is_leaf_logit, is_leaf_logit.new_tensor(gt_node.is_leaf).view(1, -1)).mean()

            # train the current node box to gt
            box = self.box_decoder(node_latent)
            box_loss = self.boxLossEstimator1(box, gt_node.get_box_quat().view(1, -1))
            latent_loss = torch.zeros_like(box_loss)
            center_loss = torch.zeros_like(box_loss)
            scale_loss = torch.zeros_like(box_loss)
            surf_loss = torch.zeros_like(box_loss)
            acap_loss = torch.zeros_like(box_loss)

            # gather information
            child_sem_gt_labels = []
            child_sem_pred_logits = []
            child_box_gt = []
            child_box_pred = []
            child_exists_gt = torch.zeros_like(child_exists_logits)
            for i in range(len(matched_gt_idx)):
                child_sem_gt_labels.append(gt_node.children[matched_gt_idx[i]].get_semantic_id(self.Tree))
                child_sem_pred_logits.append(child_sem_logits[0, matched_pred_idx[i], :].view(1, -1))
                child_box_gt.append(gt_node.children[matched_gt_idx[i]].get_box_quat())
                child_box_pred.append(child_pred_boxes[matched_pred_idx[i], :].view(1, -1))
                child_exists_gt[:, matched_pred_idx[i], :] = 1

            # train semantic labels
            child_sem_pred_logits = torch.cat(child_sem_pred_logits, dim=0)
            child_sem_gt_labels = torch.tensor(child_sem_gt_labels, dtype=torch.int64, \
                    device=child_sem_pred_logits.device)
            semantic_loss = self.semCELoss(child_sem_pred_logits, child_sem_gt_labels)
            semantic_loss = semantic_loss.sum()

            # train unused boxes to zeros
            unmatched_boxes = []
            for i in range(num_child_parts):
                if i not in matched_pred_idx:
                    unmatched_boxes.append(child_pred_boxes[i, 3:6].view(1, -1))
            if len(unmatched_boxes) > 0:
                unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
                unused_box_loss = unmatched_boxes.pow(2).sum() * 0.01
            else:
                unused_box_loss = 0.0

            # train exist scores
            positive_counts = child_exists_gt.view(-1,1).sum(dim=0)
            nBatch = len(child_exists_gt.view(-1,1))
            pos_weight = (nBatch - positive_counts)/(positive_counts + 1e-5)
            child_exists_loss = F.binary_cross_entropy_with_logits(
                input=child_exists_logits.view(-1,1), target=child_exists_gt.view(-1,1), reduction='none', pos_weight = pos_weight)
            child_exists_loss = child_exists_loss.sum()

            # calculate children + aggregate losses
            for i in range(len(matched_gt_idx)):
                child_losses = self.node_recon_loss(\
                        child_feats[:, matched_pred_idx[i], :], gt_node.children[matched_gt_idx[i]])
                box_loss = box_loss + child_losses['box']
                center_loss = center_loss + scale_loss + child_losses['center']
                latent_loss = latent_loss + child_losses['feat']
                surf_loss = surf_loss + child_losses['surf']
                acap_loss = acap_loss + child_losses['acap']
                is_leaf_loss = is_leaf_loss + child_losses['leaf']
                child_exists_loss = child_exists_loss + child_losses['exists']
                semantic_loss = semantic_loss + child_losses['semantic']

            return {'box': box_loss + unused_box_loss, 'surf': surf_loss, 'center':center_loss, 'feat': latent_loss,
                    'leaf': is_leaf_loss, 'anchor': torch.zeros_like(box_loss), 'exists': child_exists_loss, 'semantic': semantic_loss,
                    'acap': acap_loss,
                    'edge_exists': torch.zeros_like(box_loss),
                    'sym': torch.zeros_like(box_loss), 'adj': torch.zeros_like(box_loss)}


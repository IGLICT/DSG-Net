"""
    This is the main tester script for box-shape reconstruction evaluation.
    Use scripts/eval_recon_box_ae_chair.sh to run.
"""

import os
import sys
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from config import add_eval_args
from data_snv1 import PartGraphShapesDataset, Tree
from chamfer_distance import ChamferDistance
import compute_sym
from vis_utils import output_obj2pc, output_obj
import scipy.interpolate as interpolate

sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

chamferLoss = ChamferDistance()

parser = ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args()

# load train config
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path

# load object category information
print(conf.category)
Tree.load_category_info(conf.category)
semantic_color_file = '../stats/semantics_colors/'+conf.category+'.txt'
semantic_mtl_file = '../stats/semantics_colors/'+conf.category+'.mtl'

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)
conf.max_child_num = 10
conf.geopart_feat_size = 128
conf.part_feature_size = 128
conf.part_hidden_size = 128
conf.part_geo_feat_size = 128
# load model
print(conf.model_version)
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')
print(f'max Tree Depth: {conf.max_tree_depth}')

# check if eval results already exist. If so, delete it.
if os.path.exists(os.path.join(conf.result_path, conf.exp_name)):
    response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
    if response != 'y':
        sys.exit()
    shutil.rmtree(os.path.join(conf.result_path, conf.exp_name))

# create a new directory to store eval results
os.makedirs(os.path.join(conf.result_path, conf.exp_name))

# read mesh info
conf.gpu = int(conf.device.split(":")[-1])
conf = utils.add_meshinfo2conf(conf)

# create models
encoder = models.RecursiveEncoder(conf, Tree, conf.meshinfo, variational=True, probabilistic=False, orient = False)
decoder = models.RecursiveDecoder(conf, Tree, conf.meshinfo, orient = False)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

# load pretrained model
__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)

# create dataset and data loader
data_features = ['object', 'name']
dataset = PartGraphShapesDataset(conf.data_path, conf.train_dataset, data_features, Tree, load_geo=conf.load_geo)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_feats)

# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

# load unit cube pc
unit_cube = torch.from_numpy(utils.load_pts('cube.pts')).to(device)

def boxLoss(box_feature, gt_box_feature):
    pred_box_pc = utils.transform_pc_batch_angle(unit_cube, box_feature)
    pred_reweight = utils.get_surface_reweighting_batch(box_feature[:, 3:6], unit_cube.size(0))
    gt_box_pc = utils.transform_pc_batch_angle(unit_cube, gt_box_feature)
    gt_reweight = utils.get_surface_reweighting_batch(gt_box_feature[:, 3:6], unit_cube.size(0))
    dist1, dist2 = chamferLoss(gt_box_pc, pred_box_pc)
    loss1 = (dist1 * gt_reweight).sum(dim=1) / (gt_reweight.sum(dim=1) + 1e-12)
    loss2 = (dist2 * pred_reweight).sum(dim=1) / (pred_reweight.sum(dim=1) + 1e-12)
    loss = (loss1 + loss2) / 2
    return loss

def compute_struct_diff(gt_node, pred_node):
    if gt_node.is_leaf:
        if pred_node.is_leaf:
            return 0, 0, 0, 0, 0, 0
        else:
            return len(pred_node.boxes())-1, 0, 0, pred_node.get_subtree_edge_count(), 0, 0
    else:
        if pred_node.is_leaf:
            return len(gt_node.boxes())-1, 0, gt_node.get_subtree_edge_count() * 2, 0, 0, 0
        else:
            gt_sem = set([node.label for node in gt_node.children])
            pred_sem = set([node.label for node in pred_node.children])
            intersect_sem = set.intersection(gt_sem, pred_sem)

            gt_cnodes_per_sem = dict()
            for node_id, gt_cnode in enumerate(gt_node.children):
                if gt_cnode.label in intersect_sem:
                    if gt_cnode.label not in gt_cnodes_per_sem:
                        gt_cnodes_per_sem[gt_cnode.label] = []
                    gt_cnodes_per_sem[gt_cnode.label].append(node_id)

            pred_cnodes_per_sem = dict()
            for node_id, pred_cnode in enumerate(pred_node.children):
                if pred_cnode.label in intersect_sem:
                    if pred_cnode.label not in pred_cnodes_per_sem:
                        pred_cnodes_per_sem[pred_cnode.label] = []
                    pred_cnodes_per_sem[pred_cnode.label].append(node_id)

            matched_gt_idx = []; matched_pred_idx = []; matched_gt2pred = np.zeros((conf.max_child_num), dtype=np.int32)
            for sem in intersect_sem:
                gt_boxes = torch.cat([gt_node.children[cid].get_box_quat() for cid in gt_cnodes_per_sem[sem]], dim=0).to(device)
                pred_boxes = torch.cat([pred_node.children[cid].get_box_quat() for cid in pred_cnodes_per_sem[sem]], dim=0).to(device)

                num_gt = gt_boxes.size(0)
                num_pred = pred_boxes.size(0)

                if num_gt == 1 and num_pred == 1:
                    cur_matched_gt_idx = [0]
                    cur_matched_pred_idx = [0]
                else:
                    gt_boxes_tiled = gt_boxes.unsqueeze(dim=1).repeat(1, num_pred, 1)
                    pred_boxes_tiled = pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1)
                    dmat = boxLoss(gt_boxes_tiled.view(-1, 7), pred_boxes_tiled.view(-1, 7)).view(-1, num_gt, num_pred).cpu()
                    _, cur_matched_gt_idx, cur_matched_pred_idx = utils.linear_assignment(dmat)
                # print(cur_matched_gt_idx)
                # print(cur_matched_pred_idx)

                # print(sem)
                for i in range(len(cur_matched_gt_idx)):
                    # print(gt_cnodes_per_sem[sem])
                    # print(pred_cnodes_per_sem[sem])
                    # print(gt_cnodes_per_sem[sem][cur_matched_gt_idx[i]])
                    # print(pred_cnodes_per_sem[sem][cur_matched_pred_idx[i]])
                    matched_gt_idx.append(gt_cnodes_per_sem[sem][cur_matched_gt_idx[i]])
                    matched_pred_idx.append(pred_cnodes_per_sem[sem][cur_matched_pred_idx[i]])
                    matched_gt2pred[gt_cnodes_per_sem[sem][cur_matched_gt_idx[i]]] = pred_cnodes_per_sem[sem][cur_matched_pred_idx[i]]


            struct_diff = 0.0; edge_both = 0; edge_gt = 0; edge_pred = 0;
            gt_binary_diff = 0.0; gt_binary_tot = 0;
            for i in range(len(gt_node.children)):
                if i not in matched_gt_idx:
                    struct_diff += len(gt_node.children[i].boxes())
                    edge_gt += gt_node.children[i].get_subtree_edge_count() * 2

            for i in range(len(pred_node.children)):
                if i not in matched_pred_idx:
                    struct_diff += len(pred_node.children[i].boxes())
                    edge_pred += pred_node.children[i].get_subtree_edge_count()

            for i in range(len(matched_gt_idx)):
                gt_id = matched_gt_idx[i]
                pred_id = matched_pred_idx[i]
                cur_struct_diff, cur_edge_both, cur_edge_gt, cur_edge_pred, cur_gt_binary_diff, cur_gt_binary_tot = compute_struct_diff(gt_node.children[gt_id], pred_node.children[pred_id])
                gt_binary_diff += cur_gt_binary_diff
                gt_binary_tot += cur_gt_binary_tot
                struct_diff += cur_struct_diff
                edge_both += cur_edge_both
                edge_gt += cur_edge_gt
                edge_pred += cur_edge_pred
                pred_node.children[pred_id].node_id = gt_node.children[gt_id].node_id

            if pred_node.edges is not None:
                edge_pred += len(pred_node.edges)

            if gt_node.edges is None:
                edge_gt += len(gt_node.edges) * 2
                pred_edges = np.zeros((conf.max_child_num, conf.max_child_num, len(conf.edge_types)), dtype=np.bool)
                for edge in pred_node.edges:
                    pred_part_a_id = edge['part_a']
                    pred_part_b_id = edge['part_b']
                    edge_type_id = conf.edge_types.index(edge['type'])
                    pred_edges[pred_part_a_id, pred_part_b_id, edge_type_id] = True

                for edge in gt_node.edges:
                    gt_part_a_id = edge['part_a']
                    gt_part_b_id = edge['part_b']
                    edge_type_id = conf.edge_types.index(edge['type'])
                    if gt_part_a_id in matched_gt_idx and gt_part_b_id in matched_gt_idx:
                        pred_part_a_id = matched_gt2pred[gt_part_a_id]
                        pred_part_b_id = matched_gt2pred[gt_part_b_id]
                        edge_both += pred_edges[pred_part_a_id, pred_part_b_id, edge_type_id]
                        edge_both += pred_edges[pred_part_b_id, pred_part_a_id, edge_type_id]

                        # gt edges eval
                        obb1 = pred_node.children[pred_part_a_id].box.cpu().numpy()
                        obb_quat1 = pred_node.children[pred_part_a_id].get_box_quat().cpu().numpy()
                        mesh_v1, mesh_f1 = utils.gen_obb_mesh(obb1)
                        pc1 = utils.sample_pc(mesh_v1, mesh_f1, n_points=500)
                        pc1 = torch.tensor(pc1, dtype=torch.float32, device=device)
                        obb2 = pred_node.children[pred_part_b_id].box.cpu().numpy()
                        obb_quat2 = pred_node.children[pred_part_b_id].get_box_quat().cpu().numpy()
                        mesh_v2, mesh_f2 = utils.gen_obb_mesh(obb2)
                        pc2 = utils.sample_pc(mesh_v2, mesh_f2, n_points=500)
                        pc2 = torch.tensor(pc2, dtype=torch.float32, device=device)
                        if edge_type_id == 0: # ADJ
                            dist1, dist2 = chamferLoss(pc1.view(1, -1, 3), pc2.view(1, -1, 3))
                            gt_binary_diff += (dist1.sqrt().min().item() + dist2.sqrt().min().item()) / 2
                        else: # SYM
                            if edge_type_id == 2: # TRANS_SYM
                                mat1to2, _ = compute_sym.compute_trans_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
                            elif edge_type_id == 3: # REF_SYM
                                mat1to2, _ = compute_sym.compute_ref_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
                            elif edge_type_id == 1: # ROT_SYM
                                mat1to2, _ = compute_sym.compute_rot_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
                            else:
                                assert 'ERROR: unknown symmetry type: %s' % edge['type']
                            mat1to2 = torch.tensor(mat1to2, dtype=torch.float32, device=device)
                            transformed_pc1 = pc1.matmul(torch.transpose(mat1to2[:, :3], 0, 1)) + \
                                    mat1to2[:, 3].unsqueeze(dim=0).repeat(pc1.size(0), 1)
                            dist1, dist2 = chamferLoss(transformed_pc1.view(1, -1, 3), pc2.view(1, -1, 3))
                            loss = (dist1.sqrt().mean() + dist2.sqrt().mean()) / 2
                            gt_binary_diff += loss.item()
                        gt_binary_tot += 1
            
            return struct_diff, edge_both, edge_gt, edge_pred, gt_binary_diff, gt_binary_tot

def compute_binary_diff(pred_node):
    if pred_node.is_leaf:
        return 0, 0
    else:
        binary_diff = 0; binary_tot = 0;

        # all children
        for cnode in pred_node.children:
            cur_binary_diff, cur_binary_tot = compute_binary_diff(cnode)
            binary_diff += cur_binary_diff
            binary_tot += cur_binary_tot

        # current node
        if pred_node.edges is not None:
            for edge in pred_node.edges:
                pred_part_a_id = edge['part_a']
                obb1 = pred_node.children[pred_part_a_id].box.cpu().numpy()
                obb_quat1 = pred_node.children[pred_part_a_id].get_box_quat().cpu().numpy()
                mesh_v1, mesh_f1 = utils.gen_obb_mesh(obb1)
                pc1 = utils.sample_pc(mesh_v1, mesh_f1, n_points=500)
                pc1 = torch.tensor(pc1, dtype=torch.float32, device=device)
                pred_part_b_id = edge['part_b']
                obb2 = pred_node.children[pred_part_b_id].box.cpu().numpy()
                obb_quat2 = pred_node.children[pred_part_b_id].get_box_quat().cpu().numpy()
                mesh_v2, mesh_f2 = utils.gen_obb_mesh(obb2)
                pc2 = utils.sample_pc(mesh_v2, mesh_f2, n_points=500)
                pc2 = torch.tensor(pc2, dtype=torch.float32, device=device)
                if edge['type'] == 'ADJ':
                    dist1, dist2 = chamferLoss(pc1.view(1, -1, 3), pc2.view(1, -1, 3))
                    binary_diff += (dist1.sqrt().min().item() + dist2.sqrt().min().item()) / 2
                elif 'SYM' in edge['type']:
                    if edge['type'] == 'TRANS_SYM':
                        mat1to2, _ = compute_sym.compute_trans_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
                    elif edge['type'] == 'REF_SYM':
                        mat1to2, _ = compute_sym.compute_ref_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
                    elif edge['type'] == 'ROT_SYM':
                        mat1to2, _ = compute_sym.compute_rot_sym(obb_quat1.reshape(-1), obb_quat2.reshape(-1))
                    else:
                        assert 'ERROR: unknown symmetry type: %s' % edge['type']
                    mat1to2 = torch.tensor(mat1to2, dtype=torch.float32, device=device)
                    transformed_pc1 = pc1.matmul(torch.transpose(mat1to2[:, :3], 0, 1)) + \
                            mat1to2[:, 3].unsqueeze(dim=0).repeat(pc1.size(0), 1)
                    dist1, dist2 = chamferLoss(transformed_pc1.view(1, -1, 3), pc2.view(1, -1, 3))
                    loss = (dist1.sqrt().mean() + dist2.sqrt().mean()) / 2
                    binary_diff += loss.item()
                else:
                    assert 'ERROR: unknown symmetry type: %s' % edge['type']
                binary_tot += 1

        return binary_diff, binary_tot

def random_z_scene(decoder, root_code, obj_name):
    rand_root_code = torch.randn_like(root_code.repeat(10, 1)) * np.sqrt(0.1)
    os.mkdir(os.path.join(conf.result_path, conf.exp_name, obj_name, 'random_z_scene'))
    # random_scene = []
    for id in range(len(rand_root_code)):
        rand_obj = decoder.decode_scene_from_code(z = rand_root_code[id:id+1])

        rand_output_filename = os.path.join(conf.result_path, conf.exp_name, obj_name, 'random_z_scene', 'rand' + str(id) + '.json')
        PartGraphShapesDataset.save_object(obj=rand_obj, fn=rand_output_filename)
        output_obj(rand_output_filename, semantic_color_file, semantic_mtl_file, Tree, PartGraphShapesDataset)

def inter_z_scene(decoder, root_code_A, root_code_B, obj_name):
    os.mkdir(os.path.join(conf.result_path, conf.exp_name, obj_name, 'inter_z_scene'))
    embedding_sub = torch.cat([root_code_A, root_code_B], dim =0).cpu().numpy()
    random2_intpl = interpolate.griddata(np.linspace(0, 1, len(embedding_sub) * 1), embedding_sub, np.linspace(0, 1, 20), method='linear')
    random2_intpl = torch.tensor(random2_intpl,  dtype=torch.float32).to(device = root_code_A.device)

    for id in range(len(random2_intpl)):
        inter_obj = decoder.decode_scene_from_code(z = random2_intpl[id:id+1])

        inter_output_filename = os.path.join(conf.result_path, conf.exp_name, obj_name, 'inter_z_scene', 'inter' + str(id) + '.json')
        PartGraphShapesDataset.save_object(obj=inter_obj, fn=inter_output_filename)
        output_obj(inter_output_filename, semantic_color_file, semantic_mtl_file, Tree, PartGraphShapesDataset)


# test over all test shapes
num_batch = len(dataloader)
chamfer_dists = []
structure_dists = []
edge_precisions = []
edge_recalls = []
pred_binary_diffs = []
gt_binary_diffs = []
embed_code = []

# import scipy.io as sio

# feats = sio.loadmat('bedz.mat')['name']

with torch.no_grad():
    for batch_ind, batch in enumerate(dataloader):
        obj = batch[data_features.index('object')][0]
        obj.to(device)
        obj_name = batch[data_features.index('name')][0]
        if not os.path.exists(os.path.join(conf.result_path, conf.exp_name, obj_name)):
            os.mkdir(os.path.join(conf.result_path, conf.exp_name, obj_name))

        root_code_and_kld = encoder.encode_structure(obj)
        root_code = root_code_and_kld
        # print(root_code_and_kld.shape)
        # root_code = root_code_and_kld[:, :conf.feature_size*2]
        # root_code = torch.tensor(np.array(feats[batch_ind]), dtype=torch.float32).view(1, -1).to(device = root_code.device)
        recon_obj = decoder.decode_structure(z=root_code, max_depth=conf.max_tree_depth)
        loss = decoder.structure_recon_loss(z=root_code, gt_tree=obj)
        print('[%d/%d] ' % (batch_ind, num_batch), obj_name, loss)

        # save original and reconstructed object

        orig_output_filename = os.path.join(conf.result_path, conf.exp_name, obj_name, 'orig.json')
        recon_output_filename = os.path.join(conf.result_path, conf.exp_name, obj_name, 'recon.json')
        PartGraphShapesDataset.save_object(obj=obj, fn=orig_output_filename)
        PartGraphShapesDataset.save_object(obj=recon_obj, fn=recon_output_filename)

        output_obj(orig_output_filename, semantic_color_file, semantic_mtl_file, Tree, PartGraphShapesDataset)
        output_obj(recon_output_filename, semantic_color_file, semantic_mtl_file, Tree, PartGraphShapesDataset)


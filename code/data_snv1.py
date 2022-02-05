"""
    This file defines the Hierarchy of Graph Tree class and PartNet data loader.
"""

import sys
import os
import json
import torch
import numpy as np
from torch.utils import data
from pyquaternion import Quaternion
from sklearn.decomposition import PCA
from collections import namedtuple
from utils import one_hot, tree_find
import trimesh, h5py
import kornia
import torch.nn.functional as F
import random
from scipy.spatial.transform import Rotation as R
import csv
import glob
from tqdm import tqdm

def split_path(paths):
    filepath, tempfilename = os.path.split(paths)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension

def transform_verts(verts, trans_para):

    pos = trans_para[:, 0:3]
    scale = trans_para[:, 3:6]
    dir_1 = trans_para[:, 6:9]
    dir_2 = trans_para[:, 9:12]
    verts = verts * np.repeat(scale, np.size(verts, 0), 0)

    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)

    rotmat = np.vstack([dir_1, dir_2, dir_3]).T
    verts = np.dot(rotmat, np.transpose(verts))
    verts = np.transpose(verts)

    verts = verts + np.repeat(pos, np.size(verts, 0),0)

    return verts

def translocal_part2global(tensorlist, trans_para):
    numpylist = []
    trans_para = trans_para.cpu().numpy()
    for tensor in tensorlist:
        if tensor is not None:
            v = tensor.cpu().numpy().reshape(-1, 3)
            v = transform_verts(v, trans_para)
            numpylist.append(v.tolist())

    return numpylist

# store a part hierarchy of graphs for a scene/room
class Tree(object):

    # global object category information
    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_non_leaf_sem_names = []
    num_sem = None
    root_sem = None
    leaf_geos = None
    cate_id = None


    @ staticmethod
    def load_category_info(cat):
        with open(os.path.join('../stats/part_semantics/', cat+'.txt'), 'r') as fin:
            for l in fin.readlines():
                x, y, _ = l.rstrip().split()
                x = int(x)
                Tree.part_name2id[y] = x
                Tree.part_id2name[x] = y
                Tree.part_name2cids[y] = []
                if '/' in y:
                    Tree.part_name2cids['/'.join(y.split('/')[:-1])].append(x)
        Tree.num_sem = len(Tree.part_name2id) + 1
        print(Tree.num_sem)
        for k in Tree.part_name2cids:
            Tree.part_name2cids[k] = np.array(Tree.part_name2cids[k], dtype=np.int32)
            if len(Tree.part_name2cids[k]) > 0:
                Tree.part_non_leaf_sem_names.append(k)
        Tree.root_sem = Tree.part_id2name[1]

    # store a part node in the TreeA
    class Node(object):

        def __init__(self, node_id=0, is_leaf=False, is_room = True, box=None, label=None, children=None, edges=None, full_label=None, geo=None, geo_feat=None, dggeo = None, orient = None, faces = None):
            self.is_leaf = is_leaf          # store True if the part is a leaf node
            self.is_room = is_room          # store False if the node is a object
            self.node_id = node_id          # part_id in result_after_merging.json of PartNet
            self.box = box                  # box parameter for all nodes
            self.geo = geo                  # 1 x 1000 x 3 point cloud
            self.geo_id = None              # leaf node id in all node
            self.geo_box_id = None          # leaf node box id in all node
            self.geo_feat = geo_feat        # 1 x 100 geometry feature
            self.faces = faces              # facenum x 3 face index
            self.dggeo = dggeo              # 1 x pointnum x 9 deformation geo feature
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            # self.orient = orient
            self.children = [] if children is None else children
                                            # all of its children nodes; each entry is a Node instance
            self.edges = [] if edges is None else edges
                                            # all of its children relationships;
                                            # each entry is a tuple <part_a, part_b, type, params, dist>
            """
                Here defines the edges format:
                    part_a, part_b:
                        Values are the order in self.children (e.g. 0, 1, 2, 3, ...).
                        This is an directional edge for A->B.
                        If an edge is commutative, you may need to manually specify a B->A edge.
                        For example, an ADJ edge is only shown A->B,
                        there is no edge B->A in the json file.
                    type:
                        Four types considered in StructureNet: ADJ, ROT_SYM, TRANS_SYM, REF_SYM.
                    params:
                        There is no params field for ADJ edge;
                        For ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle;
                        For TRANS_SYM edge, 0-2 translation vector;
                        For REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers,
                            3-5 unit normal direction of the reflection plane.
                    dist:
                        For ADJ edge, it's the closest distance between two parts;
                        For SYM edge, it's the chamfer distance after matching part B to part A.
            """

        def get_semantic_id(self, cls):
            return cls.part_name2id[self.full_label]

        def get_semantic_one_hot(self, cls):
            out = np.zeros((1, cls.num_sem), dtype=np.float32)
            out[0, cls.part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.box.device)

        def get_box_quat1(self):
            box = self.box.cpu().numpy().squeeze()
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir /= np.linalg.norm(xdir)
            ydir = box[9:]
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)
            rotmat = np.vstack([xdir, ydir, zdir]).T
            q = Quaternion(matrix=rotmat)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            box_quat = np.hstack([center, size, quat]).astype(np.float32)
            # print(self.box.device)
            return torch.from_numpy(box_quat).view(1, -1).to(device=self.box.device)

        def get_box_quat(self):
            box = self.box.squeeze()
            # print(box)
            center = box[:3]
            size = box[3:6]
            xdir = box[6:9]
            xdir = F.normalize(xdir.unsqueeze(0), p=2, dim=1)
            ydir = box[9:12]
            ydir = F.normalize(ydir.unsqueeze(0), p=2, dim=1)
            zdir = torch.cross(xdir[0], ydir[0])
            zdir = F.normalize(zdir.unsqueeze(0), p=2, dim=1)
            rotmat = torch.cat([xdir, ydir, zdir], dim = 0).transpose(1,0).unsqueeze(0).repeat(2,1,1)
            q1 = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotmat, eps = 1e-6)
            quat = q1[0, [3, 0, 1, 2]]
            box_quat = torch.cat([center, size, quat])
            # self.set_from_box_quat(box_quat)
            return box_quat.view(1, -1).to(device=self.box.device)

        def set_from_box_quat1(self, box_quat):
            box_quat = box_quat.cpu().detach().numpy().squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            q = Quaternion(box_quat[6], box_quat[7], box_quat[8], box_quat[9])
            rotmat = q.rotation_matrix
            box = np.hstack([center, size, rotmat[:, 0].flatten(), rotmat[:, 1].flatten()]).astype(np.float32)
            self.box = torch.from_numpy(box).view(1, -1).cuda()

        def set_from_box_quat(self, box_quat):
            box_quat = box_quat.squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            rotmat = kornia.geometry.conversions.quaternion_to_rotation_matrix(box_quat[[7, 8, 9, 6]])
            box = torch.cat([center, size, rotmat[:, 0].view(-1), rotmat[:, 1].view(-1)])
            self.box = box.view(1, -1).cuda()

        def to(self, device):
            if self.box is not None:
                self.box = self.box.to(device)
            for edge in self.edges:
                if 'params' in edge:
                    edge['params'].to(device)
            if self.geo is not None:
                self.geo = self.geo.to(device)
            if self.dggeo is not None:
                self.dggeo = self.dggeo.to(device)
            # if self.orient is not None:
            #     self.orient = self.orient.to(device)
            for child_node in self.children:
                child_node.to(device)

            return self

        def _to_str(self, level, pid, detailed=False):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + (' [LEAF] ' if self.is_leaf else '    ') + '{' + str(self.part_id) + '}'
            if detailed:
                out_str += 'Box('+';'.join([str(item) for item in self.box.numpy()])+')\n'
            else:
                out_str += '\n'

            if len(self.children) > 0:
                for idx, child in enumerate(self.children):
                    out_str += child._to_str(level+1, idx)

            if detailed and len(self.edges) > 0:
                for edge in self.edges:
                    if 'params' in edge:
                        edge = edge.copy() # so the original parameters don't get changed
                        edge['params'] = edge['params'].cpu().numpy()
                    out_str += '  |'*(level) + '  ├' + 'Edge(' + str(edge) + ')\n'

            return out_str

        def __str__(self):
            return self._to_str(0, 0)

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        def child_adjacency(self, typed=False, max_children=None):
            if max_children is None:
                adj = torch.zeros(len(self.children), len(self.children))
            else:
                adj = torch.zeros(max_children, max_children)

            if typed:
                edge_types = ['ADJ', 'ROT_SYM', 'TRANS_SYM', 'REF_SYM']

            for edge in self.edges:
                if typed:
                    edge_type_index = edge_types.index(edge['type'])
                    adj[edge['part_a'], edge['part_b']] = edge_type_index
                    adj[edge['part_b'], edge['part_a']] = edge_type_index
                else:
                    adj[edge['part_a'], edge['part_b']] = 1
                    adj[edge['part_b'], edge['part_a']] = 1

            return adj

        def geos(self, leafs_only=True):
            nodes = list(self.depth_first_traversal())
            out_geos = []; out_nodes = [];
            for node in nodes:
                if not leafs_only or node.is_leaf:
                    out_geos.append(node.geo)
                    out_nodes.append(node)
            return out_geos, out_nodes

        def boxes(self, per_node=False, leafs_only=False):
            nodes = list(reversed(self.depth_first_traversal()))
            node_boxesets = []
            boxes_stack = []
            for node in nodes:
                node_boxes = []
                for i in range(len(node.children)):
                    node_boxes = boxes_stack.pop() + node_boxes

                if node.box is not None and (not leafs_only or node.is_leaf):
                    node_boxes.append(node.box)

                if per_node:
                    node_boxesets.append(node_boxes)

                boxes_stack.append(node_boxes)

            assert len(boxes_stack) == 1

            if per_node:
                return node_boxesets, list(nodes)
            else:
                boxes = boxes_stack[0]
                return boxes

        def graph(self, leafs_only=False):
            part_boxes = []
            part_geos = []
            edges = []
            node_ids = []
            part_sems = []

            nodes = list(reversed(self.depth_first_traversal()))

            box_index_offset = 0
            for node in nodes:
                child_count = 0
                box_idx = {}
                for i, child in enumerate(node.children):
                    if leafs_only and not child.is_leaf:
                        continue

                    part_boxes.append(child.box)
                    part_geos.append(child.geo)
                    node_ids.append(child.node_id)
                    part_sems.append(child.full_label)

                    box_idx[i] = child_count+box_index_offset
                    child_count += 1

                for edge in node.edges:
                    if leafs_only and not (
                            node.children[edge['part_a']].is_leaf and
                            node.children[edge['part_b']].is_leaf):
                        continue
                    edges.append(edge.copy())
                    edges[-1]['part_a'] = box_idx[edges[-1]['part_a']]
                    edges[-1]['part_b'] = box_idx[edges[-1]['part_b']]

                box_index_offset += child_count

            return part_boxes, part_geos, edges, node_ids, part_sems

        def edge_tensors(self, edge_types, device, type_onehot=True):
            num_edges = len(self.edges)

            # get directed edge indices in both directions as tensor
            edge_indices = torch.tensor(
                [[e['part_a'], e['part_b']] for e in self.edges] + [[e['part_b'], e['part_a']] for e in self.edges],
                device=device, dtype=torch.long).view(1, num_edges*2, 2)

            # get edge type as tensor
            edge_type = torch.tensor([edge_types.index(edge['type']) for edge in self.edges], device=device, dtype=torch.long)
            if type_onehot:
                edge_type = one_hot(inp=edge_type, label_count=len(edge_types)).transpose(0, 1).view(1, num_edges, len(edge_types)).to(dtype=torch.float32)
            else:
                edge_type = edge_type.view(1, num_edges)
            edge_type = torch.cat([edge_type, edge_type], dim=1) # add edges in other direction (symmetric adjacency)

            edge_dist_dict = dict()
            for e in self.edges:
                if 'min_dist' in e.keys():
                    edge_dist_dict[(e['part_a'], e['part_b'])] = e['min_dist']
                    edge_dist_dict[(e['part_b'], e['part_a'])] = e['min_dist']

            return edge_type, edge_indices, edge_dist_dict

        def get_subtree_edge_count(self):
            cnt = 0
            if self.children is not None:
                for cnode in self.children:
                    cnt += cnode.get_subtree_edge_count()
            if self.edges is not None:
                cnt += len(self.edges)
            return cnt

    # functions for class Tree
    def __init__(self, root):
        self.root = root

    def to(self, device):
        self.root = self.root.to(device)
        return self

    def __str__(self):
        return str(self.root)

    def get_geo(self):
        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=self.root, parent_json=None, parent_child_idx=None)]
        # print('a')

        # traverse the tree, converting child nodes of each node to json
        geo_id = -1
        leaf_geo = []
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node
            if len(node.children) == 0 and node.geo_id is None:
                node.geo_id = 0
                leaf_geo.append(node.get_box_quat())
                break

            for child in node.children:
                # node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=None, parent_child_idx=None))
                if child.is_leaf or len(child.children) == 0:
                    geo_id = geo_id + 1
                    child.geo_id = geo_id
                    if child.geo is not None:
                        leaf_geo.append(child.geo)
                        continue

                    if child.box is not None:
                        leaf_geo.append(child.get_box_quat())
                        continue

        self.leaf_geos = torch.cat(leaf_geo, dim = 0).unsqueeze(0).to(self.root.box.device)

    def tree2geo(self, located_para):
        part_boxes, part_geos, edges, node_ids, part_sems = self.graph(leafs_only = True)
        return translocal_part2global(part_geos, located_para), part_sems

    def depth_first_traversal(self):
        return self.root.depth_first_traversal()

    def boxes(self, per_node=False, leafs_only=False):
        return self.root.boxes(per_node=per_node, leafs_only=leafs_only)

    def graph(self, leafs_only=False):
        return self.root.graph(leafs_only=leafs_only)

    def free(self):
        for node in self.depth_first_traversal():
            del node.geo
            del node.dggeo
            del node.geo_feat
            del node.box
            del node


# extend torch.data.Dataset class for PartNet
class PartGraphShapesDataset(data.Dataset):

    def __init__(self, root, object_list, data_features, Tree, load_geo=False):
        self.root = root
        self.data_features = data_features
        self.load_geo = load_geo
        self.Tree = Tree
        # self.current_cate_id = 0
        # self.batch_size = 32
        # self.training_cate_num = 2

        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

        self.partnet_path = os.path.join(self.root, '..', 'partnetdata1')

        # accelarte the training speed, allocate all data in cpu memory
        self.all_data_obj = []
        self.object_names_new = []
        unique_id = []
        good_id = []
        with open('./good_list.txt', 'r') as f:
            good_id = [item.rstrip() for item in f.readlines()]
        for item in tqdm(self.object_names, desc="Load Data..."):
            # print(item)
            partnet_id = item
            if partnet_id not in good_id:
                continue
            if partnet_id not in unique_id:
                obj = self.load_object(os.path.join(self.root, item +'.json'), \
                        Tree = self.Tree, load_geo=self.load_geo)
                self.all_data_obj.append(obj)
                self.object_names_new.append(item)
                unique_id.append(partnet_id)

        if 0 and load_geo:
            meshinfo = os.path.join(root, 'cube_meshinfo.mat')
            meshdata = h5py.File(meshinfo, mode = 'r')

            self.point_num = meshdata['neighbour'].shape[0]
            self.edge_index = np.array(meshdata['edge_index']).astype('int64')
            self.recon = np.array(meshdata['recon']).astype('float32')
            self.ref_V = np.array(meshdata['ref_V']).astype('float32')
            self.ref_F = np.array(meshdata['ref_F']).astype('int64')
            self.vdiff = np.array(meshdata['vdiff']).astype('float32')
            self.nb = np.array(meshdata['neighbour']).astype('float32')

    def __getitem__(self, index):
        if 'object' in self.data_features:
            # obj = self.load_object(os.path.join(self.root, self.object_names[index]+'.json'), \
            #         Tree = self.Tree, load_geo=self.load_geo)
            obj = self.all_data_obj[index]
        # cate_id = tree_find(self.category_names, self.object_names[index])
        # obj.cate_id = cate_id[0]
        data_feats = ()
        for feat in self.data_features:
            if feat == 'object':
                data_feats = data_feats + (obj,)
            elif feat == 'name':
                data_feats = data_feats + (self.object_names_new[index],)
            else:
                assert False, 'ERROR: unknow feat type %s!' % feat

        return data_feats

    def __len__(self):
        return len(self.all_data_obj)

    def get_anno_id(self, anno_id):
        obj = self.load_object(os.path.join(self.root, anno_id+'.json'), \
                Tree = self.Tree, load_geo=self.load_geo)
        return obj

    @staticmethod
    def load_object(fn, Tree, load_geo=False):
        if load_geo:
            geo_fn = fn.replace('_dhier/', '_dgeo/').replace('.json', '.npz')
            geo_data = np.load(geo_fn, mmap_mode='r', allow_pickle=True)
        # root_path, _, _ = split_path(fn)
        # partnet_path = os.path.join(root_path, '..', 'partnetdata1')

        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        leaf_geos_box = []
        leaf_geos_dg = []
        leaf_geos_pts = []
        geo_id = -1
        geo_box_id = -1
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            node = Tree.Node(
                node_id=node_json['id'],
                is_leaf=('children' not in node_json or len(node_json['children']) == 0),
                label=node_json['label'])

            # print(node_json['id'])
            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)
            if load_geo:
                # print(is_room)
                node.geo = torch.tensor(geo_data['partsV'][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                LOGR = torch.tensor(geo_data['LOGR'][node_json['id']], dtype=torch.float32).view(1, -1, 3)
                S = torch.tensor(geo_data['S'][node_json['id']], dtype=torch.float32).view(1, -1, 6)
                node.dggeo = torch.cat((LOGR, S), 2)
                node.faces = torch.tensor(geo_data['F'], dtype=torch.int32)

            if 'box' in node_json:

                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)
                geo_box_id = geo_box_id + 1
                node.geo_box_id = geo_box_id

            if 'children' in node_json:
                for ci, child in enumerate(node_json['children']):
                    stack.append(StackElement(node_json=node_json['children'][ci], parent=node, parent_child_idx=ci))

            if 'edges' in node_json:
                for edge in node_json['edges']:
                    if 'params' in edge:
                        edge['params'] = torch.from_numpy(np.array(edge['params'])).to(dtype=torch.float32)
                    node.edges.append(edge)

            if parent is None:
                root = node
                # root.full_label = root.label

                # for five cates
                # root.full_label = 'object/' + root.label
                root.full_label = root.label
                # root.lable = root.full_label.split('--')[-1]
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)
        return obj

    @staticmethod
    def save_object(obj, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj.root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.node_id,
                'type': f'{node.label if node.label is not None else ""}',
                'label' : node.full_label}

            if node.geo is not None:
                node_json['geo'] = node.geo.detach().cpu().numpy().reshape(-1).tolist()

            if node.box is not None:
                node_json['box'] = node.box.detach().cpu().numpy().reshape(-1).tolist()

            if len(node.children) > 0:
                node_json['children'] = []
            for child in node.children:
                node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=node_json, parent_child_idx=len(node_json['children'])-1))

            if len(node.edges) > 0:
                node_json['edges'] = []
            for edge in node.edges:
                node_json['edges'].append(edge)
                if 'params' in edge:
                    node_json['edges'][-1]['params'] = node_json['edges'][-1]['params'].cpu().numpy().reshape(-1).tolist()

            if parent_json is None:
                obj_json = node_json
            else:
                parent_json['children'][parent_child_idx] = node_json

        # obj_json['type'] = 'house'
        with open(fn, 'w') as f:
            json.dump(obj_json, f)
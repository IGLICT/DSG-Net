"""
    This file defines minimal Tree/Node class for the PartGraph Shapes dataset
    for part tree usage
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils import data
from collections import namedtuple
import utils
import kornia
import torch.nn.functional as F
import copy
from utils import one_hot


# store a part hierarchy of graphs for a shape
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


    # store a part node in the tree
    class Node(object):

        def __init__(self, device=None, part_id=None, label=None, full_label=None, group_id=None, group_ins_id=None, is_leaf = False, box = None, children = None, edges = None, geo=None, geo_feat=None, dggeo = None, faces = None):
            self.device = device            # device that this node lives
            self.part_id = part_id          # part_id in result_after_merging.json of PartNet
            self.group_id = group_id        # group_id is 0, 1, 2, ...; it will be the same for equivalent subtree nodes
            self.group_ins_id = group_ins_id# group_ins_id is 0, 1, 2, ... within each equivalent class
            self.label = label              # node semantic label at the current level
            self.full_label = full_label    # node semantic label from root (separated by slash)
            self.children = [] if children is None else children
                                            # initialize to be empty (no children)
            self.geo_id = None              # the index of the part pc geo array
            self.geo = geo                  # 1 x 1000 x 3 point cloud
            self.geo_feat = geo_feat        # 1 x 100 geometry feature
            self.faces = faces              # facenum x 3 face index
            self.dggeo = dggeo              # 1 x pointnum x 9 deformation geo feature
            self.is_leaf = is_leaf
            self.box = box
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
        
        def get_semantic_id(self):
            return Tree.part_name2id[self.full_label]
            
        def get_semantic_one_hot(self):
            out = np.zeros((1, Tree.num_sem), dtype=np.float32)
            out[0, Tree.part_name2id[self.full_label]] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.device)
            
        def get_group_ins_one_hot(self, max_part_per_parent):
            out = np.zeros((1, max_part_per_parent), dtype=np.float32)
            out[0, self.group_ins_id] = 1
            return torch.tensor(out, dtype=torch.float32).to(device=self.device)
        
        def get_group_ins_id(self):
            return self.group_ins_id

        def to(self, device):
            if self.box is not None and not isinstance(self.box, list):
                self.box = self.box.to(device)
            for edge in self.edges:
                if 'params' in edge:
                    edge['params'].to(device)
            if self.geo is not None:
                self.geo = self.geo.to(device)
            if self.dggeo is not None:
                self.dggeo = self.dggeo.to(device)
            for child_node in self.children:
                child_node.to(device)

            return self
        
        def set_from_box_quat(self, box_quat):
            box_quat = box_quat.squeeze()
            center = box_quat[:3]
            size = box_quat[3:6]
            rotmat = kornia.quaternion_to_rotation_matrix(box_quat[[7, 8, 9, 6]])
            box = torch.cat([center, size, rotmat[:, 0].view(-1), rotmat[:, 1].view(-1)])
            self.box = box.view(1, -1).cuda()
        
        def get_box_quat(self):
            box = self.box
            return box.unsqueeze(1).to(device=self.box.device)
        
        def get_box_quat1(self):
            box = self.box
            # print(box)
            center = box[:, :3]
            size = box[:, 3:6]
            xdir = box[:, 6:9]
            xdir = F.normalize(xdir, p=2, dim=1)
            ydir = box[:, 9:]
            ydir = F.normalize(ydir, p=2, dim=1)
            zdir = torch.cross(xdir, ydir, dim = 1)
            zdir = F.normalize(zdir, p=2, dim=1)
            rotmat = torch.cat([xdir.unsqueeze(1), ydir.unsqueeze(1), zdir.unsqueeze(1)], dim = 1).transpose(2,1).repeat(2,1,1)
            # print(rotmat.shape)
            q1 = kornia.rotation_matrix_to_quaternion(rotmat, eps = 1e-6)
            quat = q1[:box.size(0), [3, 0, 1, 2]]
            # print(quat.shape)
            box_quat = torch.cat([center, size, quat], dim = 1)
            # self.set_from_box_quat(box_quat)
            return box_quat.unsqueeze(1).to(device=self.box.device)

        def _to_str(self, level, pid):
            out_str = '  |'*(level-1) + '  ├'*(level > 0) + str(pid) + ' ' + self.label + \
                    (' [LEAF %d] ' % self.geo_id if len(self.children) == 0 else '    ') + \
                    '{part_id: %d, group_id: %d [%d], subtree_geo_ids: %s}\n' % \
                    (self.part_id, self.group_id, self.group_ins_id, str(self.subtree_geo_ids))
            for idx, child in enumerate(self.children):
                out_str += child._to_str(level+1, idx)
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

        def get_leaf_ids(self):
            leaf_ids = []
            if len(self.children) == 0:
                leaf_ids.append(self.part_id)
            else:
                for cnode in self.children:
                    leaf_ids += cnode.get_leaf_ids()
            return leaf_ids

        def mark_geo_id(self, d):
            if self.part_id in d:
                self.geo_id = d[self.part_id]
            for cnode in self.children:
                cnode.mark_geo_id(d)

        def compute_subtree_geo_ids(self):
            if len(self.children) == 0:
                self.subtree_geo_ids = [self.geo_id]
            else:
                self.subtree_geo_ids = []
                for cnode in self.children:
                    self.subtree_geo_ids += cnode.compute_subtree_geo_ids()
            return self.subtree_geo_ids

        def get_subtree_edge_count(self):
            cnt = 0
            if self.children is not None:
                for cnode in self.children:
                    cnt += cnode.get_subtree_edge_count()
            if self.edges is not None:
                cnt += len(self.edges)
            return cnt
        
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
            part_ids = []
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
                    part_ids.append(child.part_id)
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

            return part_boxes, part_geos, edges, part_ids, part_sems
        
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

            return edge_type, edge_indices

        def free(self):
            for node in self.depth_first_traversal():
                node.geo = []
                node.dggeo = []
                node.geo_feat = []
                node.box = []


    # functions for class Tree
    def __init__(self, root):
        self.root = root

    def to(self, device):
        self.root = self.root.to(device)
        return self
    
    def graph(self, leafs_only=False):
        return self.root.graph(leafs_only=leafs_only)

    @staticmethod
    def load_template(fn, device):
        with open(fn, 'r') as f:
            root_json = json.load(f)

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_json, parent=None, parent_child_idx=None)]

        root = None
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json
            if node_json['group_ins_id']>9:
                print(fn)

            node = Tree.Node(device=device,
                part_id=node_json['id'],
                group_id=node_json['group_id'],
                group_ins_id=node_json['group_ins_id'],
                label=node_json['label'],
                is_leaf=('children' not in node_json),
                geo = [],
                dggeo = [],
                box = [])

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
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        return root

# extend torch.data.Dataset class for PartNet
class PartGraphShapesDataset(data.Dataset):

    def __init__(self, data_dir, pg_dir, device, batch_size, mode='sample_by_template'):
        self.data_dir = data_dir
        self.pg_dir = pg_dir
        self.device = device
        self.batch_size = batch_size
        self.mode = mode
        # self.errid = []

        self.pg_shapes = []
        self.sample_by_shape_pgids = []
        with open(os.path.join(pg_dir, 'info.txt'), 'r') as fin:
            for i, l in enumerate(fin.readlines()):
                cur_pg_shapes = l.rstrip().split()
                self.pg_shapes.append(cur_pg_shapes)
                self.sample_by_shape_pgids += [i] * len(cur_pg_shapes)

        self.pg_templates = []
        self.pg_leaf_ids = []
        self.leaf_mappings = []
        for i in range(len(self.pg_shapes)):
            cur_pg_dir = os.path.join(pg_dir, 'pt-%d' % i)
            t = Tree.load_template(os.path.join(cur_pg_dir, 'template.json'), device)
            self.pg_templates.append(t)
            leaf_ids = t.get_leaf_ids()
            t.leaf_cnt = len(leaf_ids)
            self.pg_leaf_ids.append(leaf_ids)
            t.mark_geo_id({y: x for x, y in enumerate(self.pg_leaf_ids[i])})
            t.compute_subtree_geo_ids()

            self.leaf_mappings.append([])
            for anno_id in self.pg_shapes[i]:
                with open(os.path.join(cur_pg_dir, anno_id+'.txt'), 'r') as fin:
                    tmp_dict = dict()
                    for l in fin.readlines():
                        x, y = l.rstrip().split()
                        tmp_dict[int(x)] = int(y)
                    tmp_dict[0] = 0
                cur_leaf_mapping = [tmp_dict[x] for x in range(len(tmp_dict))]
                # cur_leaf_mapping = [tmp_dict[x] for x in self.pg_leaf_ids[i]]
                cur_leaf_mapping = np.array(cur_leaf_mapping, dtype=np.int32)
                self.leaf_mappings[i].append(cur_leaf_mapping)

            self.pg_leaf_ids[i] = np.array(self.pg_leaf_ids[i], dtype=np.int32)
        
        print('[PartGraphShapesDataset %d %s %d %d] %s %s' % (batch_size, mode, \
                len(self.pg_shapes), len(self.sample_by_shape_pgids), data_dir, pg_dir))

    def __len__(self):
        if self.mode == 'sample_by_template':
            return len(self.pg_shapes)
        elif self.mode == 'sample_by_shape':
            return len(self.sample_by_shape_pgids)
        else:
            raise ValueError('ERROR: unknown mode %s!' % self.mode)

    def get_pg_shapes(self, index):
        return self.pg_shapes[index]

    def get_pg_template(self, index):
        return self.pg_templates[index]

    def get_pg_leaf_ids(self, index):
        return self.pg_leaf_ids[index]

    def get_pg_real_pcs(self, index, num_shape):
        ids = np.random.choice(len(self.pg_shapes[index]), num_shape, replace=True)
        part_pcs = np.zeros((num_shape, len(self.pg_leaf_ids[index]), 1000, 3), dtype=np.float32)
        names = []
        for i, idx in enumerate(ids):
            geo_fn = os.path.join(self.data_dir, self.pg_shapes[index][idx] + '.npz')
            geo_data = np.load(geo_fn)['parts']
            part_pcs[i] = geo_data[self.leaf_mappings[index][idx]]
            names.append(self.pg_shapes[index][idx])
        out = torch.from_numpy(part_pcs)
        return (names, out)

    def get_pg_real_pc(self, index, j):
        j = j % len(self.pg_shapes[index])
        geo_fn = os.path.join(self.data_dir, self.pg_shapes[index][j] + '.npz')
        geo_data = np.load(geo_fn)['parts']
        part_pcs = geo_data[self.leaf_mappings[index][j]]
        out = torch.from_numpy(part_pcs)
        return self.pg_shapes[index][j], out

    def __getitem__(self, index):
        if self.mode == 'sample_by_shape':
            index = self.sample_by_shape_pgids[index]
        ids = np.random.choice(len(self.pg_shapes[index]), self.batch_size, replace=True)
        pt_template = self.get_pg_template(index)
        # pt_template.free()
        pt_template = copy.deepcopy(pt_template)
        # part_pcs = np.zeros((self.batch_size, len(self.pg_leaf_ids[index]), 1000, 3), dtype=np.float32)
        for i, idx in enumerate(ids):
            geo_fn = os.path.join(self.data_dir, self.pg_shapes[index][idx] + '.npz')
            geo_data = np.load(geo_fn)
            # print(index)
            # print(geo_data['partsV'].shape)
            # print(geo_fn)
            # print(self.leaf_mappings[index][idx])
            # print(idx)
            # print(ids)
            pt_template = self.load_object_batch(pt_template, geo_data, self.leaf_mappings[index][idx], self.device, cat = (i == (len(ids)-1)))

        return (index, pt_template, ) + (self.pg_shapes[index][ids[0]],)

    @staticmethod
    def load_object_batch(Ps_tamplate, geo_data, leaf_mappings, device, cat=False):
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        # traverse the tree, converting child nodes of each node to json
        geo_id = -1
        Ps_tamplate_batch = Ps_tamplate
        stack = [StackElement(node=Ps_tamplate_batch, parent_json=None, parent_child_idx=None)]
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node
            if len(node.children) == 0 or node.is_leaf:
                # print(len(node.geo))
                node.geo.append(torch.tensor(geo_data['partsV'][leaf_mappings[node.part_id]], dtype=torch.float32).view(1, -1, 3))
                LOGR = torch.tensor(geo_data['LOGR'][leaf_mappings[node.part_id]], dtype=torch.float32).view(1, -1, 3)
                S = torch.tensor(geo_data['S'][leaf_mappings[node.part_id]], dtype=torch.float32).view(1, -1, 6)
                node.dggeo.append(torch.cat((LOGR, S), 2))
                node.faces = torch.tensor(geo_data['F'], dtype=torch.int32)
                node.box.append(torch.from_numpy(np.array(geo_data['box_quat'][leaf_mappings[node.part_id]])).to(dtype=torch.float32).view(1, -1))
                if cat:
                    node.geo = torch.cat(node.geo, dim = 0)#.to(device=device)
                    node.dggeo = torch.cat(node.dggeo, dim = 0)#.to(device=device)
                    node.box = torch.cat(node.box, dim = 0)#.to(device=device)
                # node.geo_id = 0
                # break
            else:
                node.geo.append(torch.tensor(geo_data['partsV'][leaf_mappings[node.part_id]], dtype=torch.float32).view(1, -1, 3))
                LOGR = torch.tensor(geo_data['LOGR'][leaf_mappings[node.part_id]], dtype=torch.float32).view(1, -1, 3)
                S = torch.tensor(geo_data['S'][leaf_mappings[node.part_id]], dtype=torch.float32).view(1, -1, 6)
                node.dggeo.append(torch.cat((LOGR, S), 2))
                node.box.append(torch.from_numpy(np.array(geo_data['box_quat'][leaf_mappings[node.part_id]])).to(dtype=torch.float32).view(1, -1))
                if cat:
                    node.geo = torch.cat(node.geo, dim = 0)#.to(device=device)
                    node.dggeo = torch.cat(node.dggeo, dim = 0)#.to(device=device)
                    node.box = torch.cat(node.box, dim = 0)#.to(device=device)

            for child in node.children:
                # node_json['children'].append(None)
                stack.append(StackElement(node=child, parent_json=None, parent_child_idx=None))

        return Ps_tamplate_batch
    
    @staticmethod
    def load_object(fn, Tree):

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
                part_id=node_json['id'],
                is_leaf=('children' not in node_json),
                label=node_json['label'])
            # print(node_json['id'])
            if 'geo' in node_json.keys():
                node.geo = torch.tensor(np.array(node_json['geo']), dtype=torch.float32).view(1, -1, 3)

            if 'box' in node_json:
                node.box = torch.from_numpy(np.array(node_json['box'])).to(dtype=torch.float32)

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
                root.full_label = root.label
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx+1-len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj
    
    @staticmethod
    def save_object(obj_root, fn):

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node', 'parent_json', 'parent_child_idx'])
        stack = [StackElement(node=obj_root, parent_json=None, parent_child_idx=None)]

        obj_json = None

        # traverse the tree, converting child nodes of each node to json
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent_json = stack_elm.parent_json
            parent_child_idx = stack_elm.parent_child_idx
            node = stack_elm.node

            node_json = {
                'id': node.part_id,
                'label': f'{node.label if node.label is not None else ""}'}

            if node.geo is not None:
                node_json['geo'] = node.geo.cpu().numpy().reshape(-1).tolist()

            if node.box is not None:
                node_json['box'] = node.box.cpu().numpy().reshape(-1).tolist()

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

        with open(fn, 'w') as f:
            json.dump(obj_json, f)

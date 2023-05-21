import numpy as np
from scipy.optimize import linear_sum_assignment

def to_str(node, level):
    out_str = '  |'*(level-1) + '  ├'*(level > 0) + node['label'] + '\n'
    if 'children' in node:
        for idx, child in enumerate(node['children']):
            out_str += to_str(child, level+1)
    return out_str

def cal_subtree_size(node):
    if ('children' not in node) or (len(node['children']) == 0):
        return 1
    else:
        out = 1
        for cnode in node['children']:
            out += cal_subtree_size(cnode)
    return out


def cal_tree_edit_distance(node1, node2):
    has_no_child1 = ('children' not in node1) or (len(node1['children']) == 0)
    has_no_child2 = ('children' not in node2) or (len(node2['children']) == 0)
    
    if has_no_child1:
        if has_no_child2:
            return 0
        else:
            return cal_subtree_size(node2) - 1
    
    else:
        if has_no_child2:
            return cal_subtree_size(node1) - 1
        else:
            sem1 = set([cnode['label'] for cnode in node1['children']])
            sem2 = set([cnode['label'] for cnode in node2['children']])
            joint_sem = sem1.intersection(sem1)

            diff = 0
            for sem in joint_sem:
                cnodes1 = [cnode for cnode in node1['children'] if cnode['label'] == sem]
                cnodes2 = [cnode for cnode in node2['children'] if cnode['label'] == sem]

                dist_mat = np.zeros((len(cnodes1), len(cnodes2)), dtype=np.int32)
                for cid1 in range(len(cnodes1)):
                    for cid2 in range(len(cnodes2)):
                        dist_mat[cid1, cid2] = cal_tree_edit_distance(cnodes1[cid1], cnodes2[cid2])

                rind, cind = linear_sum_assignment(dist_mat)
                for i in range(len(rind)):
                    diff += dist_mat[rind[i], cind[i]]

                for i in range(len(cnodes1)):
                    if i not in rind:
                        diff += cal_subtree_size(cnodes1[i])
                
                for i in range(len(cnodes2)):
                    if i not in cind:
                        diff += cal_subtree_size(cnodes2[i])

            for cnode in node1['children']:
                if cnode['label'] not in joint_sem:
                    diff += cal_subtree_size(cnode)
            
            for cnode in node2['children']:
                if cnode['label'] not in joint_sem:
                    diff += cal_subtree_size(cnode)

            return diff


# if __name__ == '__main__':
#     import sys
#     import json
    
#     in1 = sys.argv[1]
#     in2 = sys.argv[2]

#     with open(in1, 'r') as fin:
#         root1 = json.load(fin)

#     with open(in2, 'r') as fin:
#         root2 = json.load(fin)
    
#     print(in1)
#     print(to_str(root1, 0), cal_subtree_size(root1))
    
#     print('\n' + in2)
#     print(to_str(root2, 0), cal_subtree_size(root2))

#     print('\nEdit Distance: ', cal_tree_edit_distance(root1, root2))


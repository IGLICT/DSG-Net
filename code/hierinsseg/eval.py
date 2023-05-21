import numpy as np
from scipy.optimize import linear_sum_assignment
from tree_edit_utils import cal_tree_edit_distance, cal_subtree_size, to_str

import sys, os,glob
import json
from natsort import natsorted

in1 = sys.argv[1]
in2 = sys.argv[2]

def diff_bet_tee(in11, in21):
    with open(in11, 'r') as fin:
        root1 = json.load(fin)

    with open(in21, 'r') as fin:
        root2 = json.load(fin)

    #print(in11)
    #print(to_str(root1, 0), cal_subtree_size(root1))

    #print('\n' + in21)
    #print(to_str(root2, 0), cal_subtree_size(root2))

    # print('\nEdit Distance: ', cal_tree_edit_distance(root1, root2))
    return cal_tree_edit_distance(root1, root2)

jslist1 = natsorted(glob.glob(os.path.join(in1,'*.json')))
jslist2 = natsorted(glob.glob(os.path.join(in2,'*.json')))

value = 0.0
for i in range(len(jslist1)):
    file1 = jslist1[i]
    val = []
    for j in range(len(jslist2)):
        file2 = jslist2[j]
        val.append(diff_bet_tee(file1, file2))
    val = np.array(val)
    value+=val.min()

print(value, value/len(jslist1))

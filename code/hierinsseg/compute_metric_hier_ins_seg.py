import os
import sys
import json
from tree_edit_utils import cal_subtree_size, cal_tree_edit_distance, to_str
from progressbar import ProgressBar

result_dir = sys.argv[1]
category = sys.argv[2]

data_to_do = []
for item in os.listdir(result_dir):
    if item.endswith('.pred.json'):
        data_to_do.append(os.path.join(result_dir, item[:-10]))
print('Total to compute: ', len(data_to_do))

bar = ProgressBar()
all_tot = 0.0; all_cnt = 0;
for i in bar(range(len(data_to_do))):
    with open(data_to_do[i]+'.gt.json', 'r') as fin:
        gt_pg = json.load(fin)
    gt_subtree_size = cal_subtree_size(gt_pg)

    with open(data_to_do[i]+'.pred.json', 'r') as fin:
        pred_pg = json.load(fin)

    cur_diff = cal_tree_edit_distance(gt_pg, pred_pg) / gt_subtree_size
    all_tot += cur_diff
    all_cnt += 1

all_tot /= all_cnt
print('%.6f' % all_tot)


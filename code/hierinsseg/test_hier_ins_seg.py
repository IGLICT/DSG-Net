import os
import sys
import json
import numpy as np
import tensorflow as tf
import model_v1 as MODEL
from collections import deque
from utils import render_pc_set

in_dir = sys.argv[1]
in_cat = sys.argv[2]

batch_size = 16
num_point = 2048
min_point_score_to_exist = 0.5
min_num_point_per_part = 5
min_conf_score_to_exist = 0.2

# load stats
pid2pname = dict()
with open('resources/%s-hier.txt'%in_cat, 'r') as fin:
    for l in fin.readlines():
        x, y, _ = l.rstrip().split()
        pid2pname[int(x)] = y.split('/')[-1]

with open('resources/%s-slots.json'%in_cat, 'r') as fin:
    data = json.load(fin)
pred_parent_arr = np.array(data['parent_arr'], dtype=np.int32)
pred_sem_arr = np.array(data['sem_arr'], dtype=np.int32)
batch_pred_sem_arr = np.tile(np.expand_dims(pred_sem_arr, 0), (batch_size, 1))
pred_slot_names = data['slot_names']
pred_num_ins = len(pred_slot_names)
idx_all = np.arange(pred_num_ins)
print('Max Pred Num Parts: %d' % pred_num_ins)

# load tf model
pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
is_training_pl = tf.placeholder(tf.bool, shape=())

# Get model
mask_pred, conf_pred, end_points = MODEL.get_model(pc_pl, pred_parent_arr, pred_num_ins, is_training_pl)

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)

# Load pretrained model
loader = tf.train.Saver()
ckpt_dir = 'resources/%s-trained-models' % in_cat
for item in os.listdir(ckpt_dir):
    if item.endswith('.meta'):
        epoch_id = item.split('.')[0].split('-')[1]
        break
loader.restore(sess, os.path.join(ckpt_dir, 'epoch-'+epoch_id+'.ckpt'))
print('Loaded successfully from %s' % os.path.join(ckpt_dir, epoch_id))

# main function
def test(pcs):
    num_data = len(pcs)
    [print(pc.shape) for pc in pcs]
    pcs = np.concatenate([np.expand_dims(pc, axis=0) for pc in pcs], axis=0)
    
    feed_dict = {pc_pl: pcs, is_training_pl: False}
    mask_pred_val, conf_pred_val = sess.run([mask_pred, conf_pred], feed_dict=feed_dict)

    out_mask = mask_pred_val > min_point_score_to_exist
    out_valid = (np.sum(out_mask, axis=-1) > min_num_point_per_part) & \
            (conf_pred_val > min_conf_score_to_exist) & \
            (batch_pred_sem_arr > 0)

    results = []; out_pcs = [];
    for i in range(num_data):
        root_node = {'pid': 0, 'label': in_cat}
        cur_pcs = [];
        q = deque()
        q.append(root_node)
        while len(q) > 0:
            cur_node = q.popleft()
            cur_pid = cur_node['pid']
            selected_idx = idx_all[pred_parent_arr == cur_pid]
            is_leaf = True
            for cid in selected_idx:
                if out_valid[i, cid]:
                    is_leaf = False
                    new_node = {'pid': int(cid), 'label': pid2pname[pred_sem_arr[cid]]}
                    if 'children' not in cur_node:
                        cur_node['children'] = []
                    cur_node['children'].append(new_node)
                    q.append(new_node)
            if is_leaf:
                cur_pcs.append(pcs[i, out_mask[i, cur_pid]])
        results.append(root_node)
        out_pcs.append(cur_pcs)
    return results, out_pcs

# test
pcs = []; outs = [];
for shape_name in os.listdir(in_dir):
    if shape_name.endswith('.npy'):
        pc = np.load(os.path.join(in_dir, shape_name))
        pcs.append(pc); outs.append(shape_name)
        if len(pcs) == batch_size:
            out_trees, out_pcs = test(pcs)
            for i in range(batch_size):
                cur_shape_name = outs[i]
                with open(os.path.join(in_dir, cur_shape_name.replace('.npy', '.pred.json')), 'w') as fout:
                    json.dump(out_trees[i], fout)
                render_pc_set(os.path.join(in_dir, cur_shape_name.replace('.npy', '.pred.png')), out_pcs[i])
            pcs = []; outs = [];

if len(pcs) > 0:
    valid_len = len(pcs)
    while len(pcs) < batch_size:
        pcs.append(pcs[0])
    out_trees, out_pcs = test(pcs)

    for i in range(valid_len):
        cur_shape_name = outs[i]
        with open(os.path.join(in_dir, cur_shape_name.replace('.npy', '.pred.json')), 'w') as fout:
            json.dump(out_trees[i], fout)
        render_pc_set(os.path.join(in_dir, cur_shape_name.replace('.npy', '.pred.png')), out_pcs[i])


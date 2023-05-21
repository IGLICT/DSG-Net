"""
Input:      a point cloud
Output:     K disjoint masks, hierarchical part proposal masks (part semantics are pre-allocated)
            (extra) per-part confidence scores, regress to the IoU between Prediction and Ground-truth
"""

import tensorflow as tf
import numpy as np
import math
import os
import sys
import tf_util
from scipy.optimize import linear_sum_assignment
from collections import deque
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point, gt_num_ins, pred_num_ins):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    gt_mask_pl = tf.placeholder(tf.float32, shape=(batch_size, gt_num_ins, num_point))
    pred_parent_arr = tf.placeholder(tf.int32, shape=(pred_num_ins))
    pred_sem_arr = tf.placeholder(tf.int32, shape=(pred_num_ins))
    pred_sem_weight = tf.placeholder(tf.float32, shape=(pred_num_ins))
    gt_parent_arr = tf.placeholder(tf.int32, shape=(batch_size, gt_num_ins))
    gt_sem_arr = tf.placeholder(tf.int32, shape=(batch_size, gt_num_ins))
    return pointclouds_pl, gt_mask_pl, pred_parent_arr, pred_sem_arr, pred_sem_weight, gt_parent_arr, gt_sem_arr


def get_model(point_cloud, pred_parent_arr, pred_num_ins, is_training, weight_decay=0.0, bn_decay=None, epsilon=1e-6):
    end_points = {}

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    l0_xyz = point_cloud
    l0_points = point_cloud

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')
    
    # instance segmentation branch
    ins_net = l0_points
    ins_net = tf_util.conv1d(ins_net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='ins/fc1', bn_decay=bn_decay)
    ins_net = tf_util.conv1d(ins_net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='ins/fc2', bn_decay=bn_decay)
    ins_net = tf_util.conv1d(ins_net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='ins/fc3', bn_decay=bn_decay)
    ins_net = tf_util.conv1d(ins_net, pred_num_ins, 1, padding='VALID', bn=False, activation_fn=None, scope='ins/fc4', bn_decay=bn_decay)
    print('ins_net: ', ins_net)      # shape: B x N x K

    ins_net = tf.transpose(ins_net, [2, 0, 1])  # shape: K x B x N

    # apply hierarchical parent-children constraints
    q = deque()
    q.append(0)

    idx_list = np.arange(pred_num_ins)

    ins_out_list = [None for i in range(pred_num_ins)]
    
    root_part = tf.Variable(np.ones((1, batch_size, num_point), dtype=np.float32), dtype=tf.float32, trainable=False)
    ins_out_list[0] = root_part

    print('Building Softmax constraints...')
    while len(q) > 0:
        cur_pid = q.popleft()
        cur_cids = idx_list[pred_parent_arr == cur_pid]
        
        if len(cur_cids) > 0:
            for cid in cur_cids:
                q.append(cid)

            cur_part_masks = tf.gather(ins_net, cur_cids)
            cur_part_masks = tf.nn.softmax(cur_part_masks, dim=0)
            cur_part_masks = tf.multiply(cur_part_masks, ins_out_list[cur_pid])
            cur_part_masks = tf.clip_by_value(cur_part_masks, epsilon, 1 - epsilon)

            for idx in range(len(cur_cids)):
                ins_out_list[cur_cids[idx]] = tf.gather(cur_part_masks, [idx])

    ins_out = tf.concat(ins_out_list, 0)    # shape: K x B x N
    print('ins_out: ', ins_out)

    mask_pred = tf.transpose(ins_out, perm=[1, 0, 2])  # B x K x N
    print('mask_pred: ', mask_pred)

    # part confidence score
    conf_net = tf.reshape(l3_points, [batch_size, -1])
    conf_net = tf_util.fully_connected(conf_net, 256, bn=True, is_training=is_training, scope='conf/fc1', bn_decay=bn_decay)
    conf_net = tf_util.fully_connected(conf_net, 256, bn=True, is_training=is_training, scope='conf/fc2', bn_decay=bn_decay)
    conf_net = tf_util.fully_connected(conf_net, pred_num_ins, bn=False, activation_fn=None, scope='conf/fc3')
    conf_net = tf.nn.sigmoid(conf_net)
    print('conf_net: ', conf_net)        # shape: B x K

    return mask_pred, conf_net, end_points


def hierarchical_hungarian_matching(mask_pred, mask_gt, pred_parent_arr, pred_sem_arr, gt_parent_arr, gt_sem_arr):
    """ Input:  mask_pred       B x K_pred x N
                mask_gt         B x K_gt x N
                pred_parent_arr K_pred
                pred_sem_arr    K_pred
                gt_parent_arr   B x K_gt
                gt_sem_arr      B x K_gt
        Output: matching_idx    B x K_pred
    """
    epsilon = 1e-4

    batch_size = mask_pred.shape[0]
    pred_num_ins = mask_pred.shape[1]
    gt_num_ins = mask_gt.shape[1]

    matching_idx = -np.ones((batch_size, pred_num_ins), dtype=np.int32)
    for i in range(batch_size):
        cur_mask_pred = mask_pred[i]
        cur_mask_gt = mask_gt[i]
        cur_gt_parent_arr = gt_parent_arr[i]
        cur_gt_sem_arr = gt_sem_arr[i]

        # start hierarchical matching
        q = deque()
        q.append((0, 0))    # match gt part 0 to pred part 0
        gt_idxs = np.arange(gt_num_ins)
        pred_idxs = np.arange(pred_num_ins)
        while len(q) > 0:
            gt_pid, pred_pid = q.popleft()

            gt_cids = gt_idxs[cur_gt_parent_arr == gt_pid]
            pred_cids = pred_idxs[pred_parent_arr == pred_pid]
            
            if len(gt_cids) > 0:
                gt_sem2pids = dict()
                for j in gt_cids:
                    sem_id = cur_gt_sem_arr[j]
                    if str(sem_id) not in gt_sem2pids.keys():
                        gt_sem2pids[str(sem_id)] = []
                    gt_sem2pids[str(sem_id)].append(j)
                
                pred_sem2pids = dict()
                for j in pred_cids:
                    sem_id = pred_sem_arr[j]
                    if str(sem_id) not in pred_sem2pids.keys():
                        pred_sem2pids[str(sem_id)] = []
                    pred_sem2pids[str(sem_id)].append(j)

                for sem_id_str in gt_sem2pids.keys():
                    sem_id = int(sem_id_str)
                    if sem_id == 0:
                        matching_idx[i, pred_sem2pids['0'][0]] = gt_sem2pids['0'][0]
                    else:
                        cur_gt_pids = np.array(gt_sem2pids[sem_id_str], dtype=np.int32)
                        cur_pred_pids = np.array(pred_sem2pids[sem_id_str], dtype=np.int32)
                        cur_gt_part_masks = cur_mask_gt[cur_gt_pids, :]
                        cur_pred_part_masks = cur_mask_pred[cur_pred_pids, :]
                        num_gt_parts = len(cur_gt_pids)
                        num_pred_parts = len(cur_pred_pids)
                        matching_score = np.matmul(cur_gt_part_masks, np.transpose(cur_pred_part_masks))
                        matching_score = 1 - np.divide(matching_score + epsilon, \
                                np.tile(np.expand_dims(np.sum(cur_gt_part_masks, 1), -1), (1, num_pred_parts)) + \
                                np.tile(np.expand_dims(np.sum(cur_pred_part_masks, 1), 0), (num_gt_parts, 1)) - matching_score + epsilon)
                        row_ind, col_ind = linear_sum_assignment(matching_score)
                        for j in range(len(row_ind)):
                            matched_gt_pid = cur_gt_pids[row_ind[j]]
                            matched_pred_pid = cur_pred_pids[col_ind[j]]
                            matching_idx[i, matched_pred_pid] = matched_gt_pid
                            q.append((matched_gt_pid, matched_pred_pid))
    
    return matching_idx


def get_ins_loss(mask_pred, mask_gt, pred_parent_arr, pred_sem_arr, pred_sem_weight, gt_parent_arr, gt_sem_arr, end_points, epsilon=1e-6):
    """ Input:  mask_pred       B x K_pred x N
                mask_gt         B x K_gt x N
                pred_parent_arr K_pred
                pred_sem_arr    K_pred
                pred_sem_weight K_pred
                gt_parent_arr   B x K_gt
                gt_sem_arr      B x K_gt
    """
    batch_size = mask_pred.get_shape()[0].value
    num_point = mask_pred.get_shape()[2].value
    pred_num_ins = mask_pred.get_shape()[1].value
    gt_num_ins = mask_gt.get_shape()[1].value

    # compute matching_idx
    matching_idx = tf.py_func(hierarchical_hungarian_matching, [mask_pred, mask_gt, pred_parent_arr, pred_sem_arr, gt_parent_arr, gt_sem_arr], tf.int32)   # B x K_pred
    matching_idx = tf.stop_gradient(matching_idx)
    end_points['matching_idx'] = matching_idx

    # gather all matched pred masks
    idx = tf.cast(tf.where(tf.greater_equal(matching_idx, 0)), tf.int32)   # -1 x 2
    pred_mask_matched = tf.gather_nd(mask_pred, idx)    # -1 x N
    end_points['pred_gather_idx'] = idx

    # gather all matched gt masks
    shape_idx = idx[:, 0]
    gt_idx = tf.gather_nd(matching_idx, idx)
    shape_gt_idx = tf.concat((tf.expand_dims(shape_idx, -1), tf.expand_dims(gt_idx, -1)), 1)  # -1 x 2
    gt_mask_matched = tf.gather_nd(mask_gt, shape_gt_idx)    # -1 x N
    end_points['gt_gather_idx'] = shape_gt_idx

    # compute soft iou
    matching_score = tf.reduce_sum(tf.multiply(gt_mask_matched, pred_mask_matched), 1)
    iou_all = tf.divide(matching_score + epsilon, tf.reduce_sum(gt_mask_matched, 1) + tf.reduce_sum(pred_mask_matched, 1) - matching_score + epsilon)
    end_points['iou_all'] = iou_all

    # divide by per-shape matched counts
    weight_matrix = tf.tile(tf.reduce_sum(tf.cast(tf.greater_equal(matching_idx, 0), tf.float32), axis=1, keep_dims=True), [1, pred_num_ins])  # B x K_pred
    weight_all = tf.gather_nd(weight_matrix, idx)
    end_points['weight_all'] = weight_all

    # balance weights for different part semantics
    sem_weight_matrix = tf.tile(tf.expand_dims(pred_sem_weight, 0), [batch_size, 1])   # B x K_pred
    sem_weight_all = tf.gather_nd(sem_weight_matrix, idx)
    end_points['sem_weight_all'] = sem_weight_all

    iou_all_weighted = tf.div(iou_all, weight_all)
    iou_all_weighted = tf.multiply(iou_all_weighted, sem_weight_all)
    loss = -tf.reduce_sum(iou_all_weighted) / batch_size

    return loss, end_points


def get_conf_loss(conf_pred, end_points):
    """ Input:  conf_pred       B x K
    """
    batch_size = conf_pred.get_shape()[0].value
    nmask = conf_pred.get_shape()[1].value

    conf_target = tf.scatter_nd(end_points['pred_gather_idx'], end_points['iou_all'], tf.constant([batch_size, nmask]))
    end_points['conf_target'] = conf_target

    per_part_loss = tf.squared_difference(conf_pred, conf_target)
    end_points['per_part_loss'] = per_part_loss

    target_pos_mask = tf.cast(tf.greater(conf_target, 0.1), tf.float32)
    target_neg_mask = 1.0 - target_pos_mask
    
    pos_per_shape_loss = tf.divide(tf.reduce_sum(target_pos_mask * per_part_loss, axis=-1), tf.maximum(1e-6, tf.reduce_sum(target_pos_mask, axis=-1)))
    neg_per_shape_loss = tf.divide(tf.reduce_sum(target_neg_mask * per_part_loss, axis=-1), tf.maximum(1e-6, tf.reduce_sum(target_neg_mask, axis=-1)))

    per_shape_loss = pos_per_shape_loss + neg_per_shape_loss
    end_points['per_shape_conf_loss'] = per_shape_loss

    loss = tf.reduce_mean(per_shape_loss)
    return loss, end_points


def get_l21_norm(mask_pred, end_points):
    """ Input:  mask_pred           B x K x N
    """
    num_point = mask_pred.get_shape()[2].value
    per_shape_l21_norm = tf.norm(tf.norm(mask_pred + 1e-12, ord=2, axis=-1), ord=1, axis=-1) / num_point
    print('per_shape_l21_norm: ', per_shape_l21_norm)
    end_points['per_shape_l21_norm'] = per_shape_l21_norm

    loss = tf.reduce_mean(per_shape_l21_norm)
    return loss, end_points



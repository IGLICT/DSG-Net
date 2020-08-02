"""
    This file contains all helper utility functions.
"""

import os
import sys
import math
import importlib
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import trimesh, configparser
from pyquaternion import Quaternion
import h5py

def worker_init_fn(worker_id):
    """ The function is designed for pytorch multi-process dataloader.
        Note that we use the pytorch random generator to generate a base_seed.
        Please try to be consistent.
        References:
            https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    #print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)

def save_checkpoint(models, model_names, dirname, epoch=None, prepend_epoch=False, optimizers=None, optimizer_names=None):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        torch.save(model.state_dict(), os.path.join(dirname, filename))

    if optimizers is not None:
        filename = 'checkpt.pth'
        if prepend_epoch:
            filename = f'{epoch}_' + filename
        checkpt = {'epoch': epoch}
        for opt, optimizer_name in zip(optimizers, optimizer_names):
            checkpt[f'opt_{optimizer_name}'] = opt.state_dict()
        torch.save(checkpt, os.path.join(dirname, filename))

def load_checkpoint(models, model_names, dirname, epoch=None, device="cuda:0", optimizers=None, optimizer_names=None, strict=True):
    if len(models) != len(model_names) or (optimizers is not None and len(optimizers) != len(optimizer_names)):
        raise ValueError('Number of models, model names, or optimizers does not match.')

    for model, model_name in zip(models, model_names):
        filename = f'net_{model_name}.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        state_dict = torch.load(os.path.join(dirname, filename), map_location=torch.device(device))
        current_state_dict = model.state_dict()
        firstkey = [k for k in current_state_dict.keys()]
        if firstkey[0].find('module.')>=0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            # state_dict = torch.load(os.path.join(dirname, filename))
            for k, v in state_dict.items():
                name = 'module.' + k # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)

    start_epoch = 0
    if optimizers is not None:
        filename = 'checkpt.pth'
        if epoch is not None:
            filename = f'{epoch}_' + filename
        filename = os.path.join(dirname, filename)
        if os.path.exists(filename):
            checkpt = torch.load(filename)
            start_epoch = checkpt['epoch']
            for opt, optimizer_name in zip(optimizers, optimizer_names):
                opt.load_state_dict(checkpt[f'opt_{optimizer_name}'])
            print(f'resuming from checkpoint {filename}')
        else:
            response = input(f'Checkpoint {filename} not found for resuming, refine saved models instead? (y/n) ')
            if response != 'y':
                sys.exit()

    return start_epoch

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def optimizer_to_device1(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(device)

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    m = rotvector.new_tensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
    return m

def get_model_module(model_version):
    importlib.invalidate_caches()
    return importlib.import_module(model_version)

# row_counts, col_counts: row and column counts of each distance matrix (assumed to be full if given)
def linear_assignment(distance_mat, row_counts=None, col_counts=None):
    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(distance_mat.shape[0]):
        # print(f'{i} / {distance_mat.shape[0]}')

        dmat = distance_mat[i, :, :]
        if row_counts is not None:
            dmat = dmat[:row_counts[i], :]
        if col_counts is not None:
            dmat = dmat[:, :col_counts[i]]

        rind, cind = linear_sum_assignment(dmat.to('cpu').detach().numpy())
        rind = list(rind)
        cind = list(cind)
        # print(dmat)
        # print(rind)
        # print(cind)

        if len(rind) > 0:
            rind, cind = zip(*sorted(zip(rind, cind)))
            rind = list(rind)
            cind = list(cind)

        # complete the assignemnt for any remaining non-active elements (in case row_count or col_count was given),
        # by assigning them randomly
        #if len(rind) < distance_mat.shape[1]:
        #    rind.extend(set(range(distance_mat.shape[1])).difference(rind))
        #    cind.extend(set(range(distance_mat.shape[1])).difference(cind))

        batch_ind += [i]*len(rind)
        row_ind += rind
        col_ind += cind

    return batch_ind, row_ind, col_ind

def object_batch_boxes(objects, max_box_num):
    box_num = []
    boxes = torch.zeros(len(objects), 12, max_box_num)
    for oi, obj in enumerate(objects):
        obj_boxes = obj.boxes()
        box_num.append(len(obj_boxes))
        if box_num[-1] > max_box_num:
            print(f'WARNING: too many boxes in object, please use a dataset that does not have objects with too many boxes, clipping the object for now.')
            box_num[-1] = max_box_num
            obj_boxes = obj_boxes[:box_num[-1]]
        obj_boxes = [o.view(-1, 1) for o in obj_boxes]
        boxes[oi, :, :box_num[-1]] = torch.cat(obj_boxes, dim=1)

    return boxes, box_num

# out shape: (label_count, in shape)
def one_hot(inp, label_count):
    out = torch.zeros(label_count, inp.numel(), dtype=torch.uint8, device=inp.device)
    out[inp.view(-1), torch.arange(out.shape[1])] = 1
    out = out.view((label_count,) + inp.shape)
    return out

def collate_feats(b):
    return list(zip(*b))

def export_ply_with_label(out, v, l):
    num_colors = len(colors)
    with open(out, 'w') as fout:
        fout.write('ply\n')
        fout.write('format ascii 1.0\n')
        fout.write('element vertex '+str(v.shape[0])+'\n')
        fout.write('property float x\n')
        fout.write('property float y\n')
        fout.write('property float z\n')
        fout.write('property uchar red\n')
        fout.write('property uchar green\n')
        fout.write('property uchar blue\n')
        fout.write('end_header\n')

        for i in range(v.shape[0]):
            cur_color = colors[l[i]%num_colors]
            fout.write('%f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], \
                    int(cur_color[0]*255), int(cur_color[1]*255), int(cur_color[2]*255)))

def load_pts(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        pts = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2])] for line in lines], dtype=np.float32)
        return pts

def export_pts(out, v):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('%f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def color2mtl(colorfile):
    from vis_utils import load_semantic_colors
    from datav1 import Tree
    filepath, fullflname = os.path.split(colorfile)
    fname, ext = os.path.splitext(fullflname)
    Tree.load_category_info(fname)
    
    sem_colors = load_semantic_colors(filename=colorfile)
    for sem in sem_colors:
        sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)

    mtl_fid = open(os.path.join(filepath, fname + '.mtl'), 'w')
    for i in range(len(Tree.part_id2name)):
        partname = Tree.part_id2name[i + 1]
        color = sem_colors[partname]
        mtl_fid.write('newmtl m_%s\nKd %f %f %f\nKa 0 0 0\n' % (partname.replace('/', '-'), color[0], color[1], color[2]))
    mtl_fid.close()

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

# pc is N x 3, feat is 10-dim
def transform_pc(pc, feat):
    num_point = pc.size(0)
    center = feat[:3]
    shape = feat[3:6]
    quat = feat[6:]
    pc = pc * shape.repeat(num_point, 1)
    pc = qrot(quat.repeat(num_point, 1), pc)
    pc = pc + center.repeat(num_point, 1)
    return pc

# pc is N x 3, feat is B x 10-dim
def transform_pc_batch(pc, feat, anchor=False):
    batch_size = feat.size(0)
    num_point = pc.size(0)
    pc = pc.repeat(batch_size, 1, 1)
    center = feat[:, :3].unsqueeze(dim=1).repeat(1, num_point, 1)
    shape = feat[:, 3:6].unsqueeze(dim=1).repeat(1, num_point, 1)
    quat = feat[:, 6:].unsqueeze(dim=1).repeat(1, num_point, 1)
    if not anchor:
        pc = pc * shape
    pc = qrot(quat.view(-1, 4), pc.view(-1, 3)).view(batch_size, num_point, 3)
    if not anchor:
        pc = pc + center
    return pc

def get_surface_reweighting(xyz, cube_num_point):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).repeat(np*2), (y*z).repeat(np*2), (x*z).repeat(np*2)])
    out = out / (out.sum() + 1e-12)
    return out

def get_surface_reweighting_batch(xyz, cube_num_point):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    assert cube_num_point % 6 == 0, 'ERROR: cube_num_point %d must be dividable by 6!' % cube_num_point
    np = cube_num_point // 6
    out = torch.cat([(x*y).unsqueeze(dim=1).repeat(1, np*2), \
                     (y*z).unsqueeze(dim=1).repeat(1, np*2), \
                     (x*z).unsqueeze(dim=1).repeat(1, np*2)], dim=1)
    out = out / (out.sum(dim=1).unsqueeze(dim=1) + 1e-12)
    return out

def gen_obb_mesh(obbs):
    # load cube
    cube_v, cube_f = load_obj('cube.obj')

    all_v = []; all_f = []; vid = 0;
    for pid in range(obbs.shape[0]):
        p = obbs[pid, :]
        center = p[0: 3]
        lengths = p[3: 6]
        dir_1 = p[6: 9]
        dir_2 = p[9: ]

        dir_1 = dir_1/np.linalg.norm(dir_1)
        dir_2 = dir_2/np.linalg.norm(dir_2)
        dir_3 = np.cross(dir_1, dir_2)
        dir_3 = dir_3/np.linalg.norm(dir_3)

        v = np.array(cube_v, dtype=np.float32)
        f = np.array(cube_f, dtype=np.int32)
        rot = np.vstack([dir_1, dir_2, dir_3])
        v *= lengths
        v = np.matmul(v, rot)
        v += center

        all_v.append(v)
        all_f.append(f+vid)
        vid += v.shape[0]

    all_v = np.vstack(all_v)
    all_f = np.vstack(all_f)
    return all_v, all_f

def sample_pc(v, f, n_points=2048):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def argpaser2file(args, name='example.ini'):
    d = args.__dict__
    cfpar = configparser.ConfigParser()
    cfpar['default'] = {}
    for key in sorted(d.keys()):
        cfpar['default'][str(key)]=str(d[key])
        print('%s = %s'%(key,d[key]))

    with open(name, 'w') as configfile:
        cfpar.write(configfile)

def inifile2args(args, ininame='example.ini'):

    config = configparser.ConfigParser()
    config.read(ininame)
    defaults = config['default']
    result = dict(defaults)
    # print(result)
    # print('\n')
    # print(args)
    args1 = vars(args)
    # print(args1)

    args1.update({k: v for k, v in result.items() if v is not None})  # Update if v is not None

    # print(args1)
    args.__dict__.update(args1)

    # print(args)

    return args

def neighbour2vdiff(neighbour, ref_V):
    neighbour = neighbour - 1
    pointnum = neighbour.shape[0]
    maxdim = neighbour.shape[1]
    vdiff = np.zeros((pointnum, maxdim, 3), dtype=np.float32)
    for point_i in range(pointnum):
        for j in range(maxdim):
            curneighbour = neighbour[point_i][j]
            if curneighbour == -1:
                break

            vdiff[point_i][j] = ref_V[point_i] - ref_V[curneighbour]

    return vdiff

# ----------------------------------------------------------- ICP -------------------------------------------------------------

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    #print(A.shape, B.shape)
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class MeshDataLoader():
    def __init__(self, data_path):

        meshfile = os.path.join(data_path, '..', 'cube_meshinfo.mat')
        if os.path.exists(meshfile):
            print(meshfile)
            meshdata = h5py.File(meshfile, mode = 'r')

            self.num_point = meshdata['neighbour'].shape[0]
            self.edge_index = np.array(meshdata['edge_index']).astype('int64')
            self.reconmatrix = np.array(meshdata['recon']).astype('float32')
            self.ref_V = np.array(meshdata['ref_V']).astype('float32')
            self.ref_F = np.array(meshdata['ref_F']).astype('int64')
            self.vdiff = np.array(meshdata['vdiff']).astype('float32')
            self.nb = np.array(meshdata['neighbour']).astype('int64')

        else:
            meshfile = os.path.join(data_path, '..', 'cube_meshinfo.npz')
            print(meshfile)
            meshinfo = np.load(meshfile)

            self.num_point = meshinfo['neighbour'].shape[0]
            self.edge_index = meshinfo['edge_index']
            self.reconmatrix = meshinfo['reconmatrix']
            self.ref_V = meshinfo['ref_V']
            self.ref_F = meshinfo['ref_F']
            self.vdiff = meshinfo['vdiff']
            self.nb = meshinfo['neighbour']

        self.part_num = 0
        self.avg_feat = 0

        print(self.num_point)
        print(self.edge_index.shape)
        print(self.reconmatrix.shape)
        print(self.ref_V.shape)
        print(self.ref_F.shape)
        print(self.vdiff.shape)
        print(self.nb.shape)
        print('-------------------------')

def add_meshinfo2conf(conf):

    conf.meshinfo = MeshDataLoader(conf.data_path)

    conf.meshinfo.edge_index = torch.tensor(conf.meshinfo.edge_index).to(conf.device)
    conf.meshinfo.reconmatrix = torch.tensor(conf.meshinfo.reconmatrix).to(conf.device)
    conf.meshinfo.ref_V = torch.tensor(conf.meshinfo.ref_V).to(conf.device)
    conf.meshinfo.ref_F = conf.meshinfo.ref_F
    conf.meshinfo.vdiff = torch.tensor(conf.meshinfo.vdiff).to(conf.device)
    conf.meshinfo.nb = torch.tensor(conf.meshinfo.nb).to(conf.device)
    conf.meshinfo.gpu = conf.gpu
    conf.meshinfo.point_num = conf.meshinfo.ref_V.shape[0]

    return conf

def tree_find(tree, value):
    def tree_rec(tree, iseq):
        if isinstance(tree, list):
            for i, child in enumerate(tree):
                r = tree_rec(child, iseq + [i])
                if r is not None:
                    return r
        elif tree == value:
            return iseq
        else:
            return None

    return tree_rec(tree, [])

def weights_init(m):
    if hasattr(m, 'weight'):
        m.weight.data.fill_(0)#normal_(0.0, 0.000001)
    if hasattr(m, 'bias'):
        m.bias.data.fill_(0)# + 1e-6

def backup_code(path = '../data/codebak'):
    import time, random
    timecurrent = time.strftime('%m%d%H%M', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
    print("code backuping... " + timecurrent, end=' ', flush=True)
    os.makedirs(os.path.join(path, timecurrent))
    os.system('cp -r ./* %s/\n' % os.path.join(path, timecurrent))
    print("DONE!")



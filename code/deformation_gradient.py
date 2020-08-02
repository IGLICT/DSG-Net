from __future__ import print_function
import numpy as np
import torch
import sys
import os
import pymesh
import math
import glob

import torch.nn.functional as F
from torch_scatter import scatter_add
from numpy.linalg import inv, svd, det, norm
from scipy.linalg import polar
from natsort import natsorted, ns
import utils
import shutil
from shutil import copyfile
import json
from pyquaternion import Quaternion

file1 = sys.argv[1]
cate1 = sys.argv[2]

class Deform_Gradiant():
    # A class for store the deformation gradient for one object
    def __init__(self, num_points):
        self.LOGR = np.zeros((num_points, 9))
        self.S = np.zeros((num_points, 9))

    def add(self, vi, LOGR, S):
        self.LOGR[vi] = np.reshape(LOGR.transpose(), (1, 9))
        self.S[vi] = np.reshape(S, (1, 9))


def GenerateMeshNormals(pos, face):
    # assert 'face' in data
    # pos, face = data.pos, data.face
    # numpy format, not the torch version

    face = torch.tensor(face, dtype = torch.int64)
    pos = torch.tensor(pos, dtype = torch.float32)
    vec1 = pos[:, face[:, 1]] - pos[:, face[:, 0]]
    vec2 = pos[:, face[:, 2]] - pos[:, face[:, 0]]

    face_norm = F.normalize(torch.cross(vec1, vec2, dim = 2), p=2, dim=-1)  # [B, F, 3]

    idx = torch.cat([face[:, 0], face[:, 1], face[:, 2]], dim=0)#.cuda()
    face_norm = face_norm.repeat(1, 3, 1)
    norm = scatter_add(face_norm, idx, dim=1)

    norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]
    # print(norm.shape)
    # data.norm = norm
    return norm

def cotangent(V, F):
    """
    Numpy format
    Input:
      V: 1 x N x 3
      F: 1 x F  x3
    Outputs:
      C: 1 x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    1 x F x 3 x 3
    """
    # F = torch.tensor(F, dtype = torch.int64)
    indices_repeat = np.stack([F, F, F], axis=2)
    # print(indices_repeat.shape)

    #v1 is the list of first triangles B*F*3, v2 second and v3 third
    # v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    # v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    # v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())
    v2 = np.take_along_axis(V, indices_repeat[:, :, :, 1], axis = 1)
    v1 = np.take_along_axis(V, indices_repeat[:, :, :, 0], axis = 1)
    v3 = np.take_along_axis(V, indices_repeat[:, :, :, 2], axis = 1)
    # print(v1.shape)
    # print(v2.shape)
    # print(v3.shape)

    l1 = np.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
    l2 = np.sqrt(((v3 - v1)**2).sum(2))
    l3 = np.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
    A = 2*np.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))
    # print(A)

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    C = np.stack([cot23, cot31, cot12], axis = 2) / np.expand_dims(A, axis = 2) / 4

    return C

def sexp(x):
    if x<=0:
        return np.exp(x)
    else:
        return 1+x

def construct_topo(ref_points, ref_Face):
    """
    Numpy format
    Input:
      V: N x 3
      F: F x 3
    Outputs:
      Two Dicts: neighbours of each vertices and cotweight of each edge, two
                 ditcs has same size
    """
    cot_weight = cotangent(np.expand_dims(ref_points, axis = 0), np.expand_dims(ref_Face, axis = 0))[0]
    cot_weight = cot_weight[:, [2, 0, 1]]
    neighbors_dict = {}
    cotweight_dict_tmp = {}
    cotweight_dict = {}
    for face_id, (face) in enumerate(ref_Face):
        for eid, e in enumerate([(0, 1), (1, 2), (2, 0)]):
            e1 = face[e[0]]
            e2 = face[e[1]]
            edge = tuple(sorted((e1, e2)))
            if edge not in cotweight_dict_tmp.keys():
                cotweight_dict_tmp[edge] = []
                cotweight_dict_tmp[edge].append(cot_weight[face_id, eid])
            else:
                cotweight = cotweight_dict_tmp[edge] + cot_weight[face_id, eid]
                if math.isnan(cotweight) or math.isinf(cotweight) or cotweight > 100000 or cotweight < 1e-3:
                    cotweight = [1.0]
                # print(cotweight)
                cotweight_dict_tmp[edge] = cotweight

    for face_id, (face) in enumerate(ref_Face):
        for eid, e in enumerate([(0, 1), (1, 2), (2, 0)]):
            e1 = face[e[0]]
            e2 = face[e[1]]
            edge = tuple(sorted((e1, e2)))
            if e1 not in neighbors_dict.keys():
                neighbors_dict[e1] = []
                neighbors_dict[e1].append(e2)
                cotweight_dict[e1] = []
                cotweight_dict[e1].append(cotweight_dict_tmp[edge])
            else:
                if e2 not in neighbors_dict[e1]:
                    neighbors_dict[e1].append(e2)
                    cotweight_dict[e1].append(cotweight_dict_tmp[edge])

    return neighbors_dict, cotweight_dict

def log_matrix(matrix):
    # compute the Logarithm of the matrix
    theta = np.maximum(-1.0, np.minimum(1.0, (np.trace(matrix) - 1) / 2.0))
    theta = np.arccos(theta)
    if np.abs(theta) <= 1e-3:
        return np.zeros((3, 3))
    else:
        if norm(matrix - matrix.transpose()) <= 1e-6:
            return np.zeros((3, 3))
        else:
            return theta / (2 * np.sin(theta)) * (matrix - matrix.transpose())

def exp_matrix(logr33):
    # compute the exp of the matrix
    thetav = np.array([logr33[0], logr33[1], logr33[2]])
    matrix = np.array([[0, logr33[0], logr33[1]], [-logr33[0], 0, logr33[3]], [-logr33[1], -logr33[3], 0]])
    theta = np.linalg.norm(thetav)
    if np.abs(theta) == 0:
        r = np.eye(3)
    else:
        x = matrix / theta
        r = np.eye(3) + x * np.sin(theta) + (1 - np.cos(theta)) * x * x

    return r

def polar_decomposition(matrix):
    u, p = polar(matrix)
    r, s = u, p
    return r, s

def polar_dec(matrix):
    u, s, vh = svd(matrix, full_matrices = False)
    vh = vh.transpose()
    R = np.matmul(u, vh.transpose())
    S = np.matmul(vh, np.matmul(np.diag(s), vh.transpose()))
    # determinate of R must be large than Zero
    if det(R) < 0:
        mins_id = np.argmin(s)
        u[:, mins_id] *= -1
        s[mins_id] *= -1
        R = np.matmul(u, vh.transpose())
        S = np.matmul(vh, np.matmul(np.diag(s), vh.transpose()))

    return log_matrix(R), S

def AffineAlign(patch_ref, patch_deform):
    # calculate the affine transformation of each correspond patch
    patch_ref = np.array(patch_ref, dtype = np.float32)
    patch_deform = np.array(patch_deform, dtype = np.float32)
    AtA = np.matmul(patch_ref.transpose(), patch_ref)
    # print(AtA)
    AtA_inv = inv(AtA)

    B = np.matmul(patch_ref.transpose(), patch_deform)
    Affine_matrix = np.matmul(AtA_inv, B)

    return Affine_matrix


def get_deforma_gradient(ref_points, points, ref_Face):
    # calculate the deformation gradients for point cloud
    normalScale = 0.3

    assert(len(ref_points)==len(points))
    # print(len(ref_points))
    # print(len(points))
    neighbors_ref, cotweight_ref = construct_topo(ref_points, ref_Face)

    normals_ref = GenerateMeshNormals(np.expand_dims(ref_points, axis = 0), ref_Face).numpy()[0]
    normals_points = GenerateMeshNormals(np.expand_dims(points, axis = 0), ref_Face).numpy()[0]
    # calculate the local deformation gradient of each patch (1-ring neighbours)

    deformation_gradient = Deform_Gradiant(len(points))
    for vi in range(len(points)):
        neighbor = neighbors_ref[vi]
        ref_patch = []
        deform_patch = []
        lens_ref = 0.0
        lens = 0.0
        for vvi_id, vvi in enumerate(neighbor):
            cot_weight_edge = cotweight_ref[vi][vvi_id]
            q_ref = (ref_points[vvi, :] - ref_points[vi, :]) * cot_weight_edge
            q = (points[vvi, :] -  points[vi, :]) * cot_weight_edge
            ref_patch.append(q_ref)
            deform_patch.append(q)
            lens_ref += norm(q_ref)
            lens += norm(q)
        ref_patch.append(normals_ref[vi] * (lens_ref / len(ref_patch) * normalScale))
        deform_patch.append(normals_points[vi] * (lens / len(deform_patch) * normalScale))
        # svd decomposition to get the deformation gradients

        Affine_matrix = AffineAlign(ref_patch, deform_patch)
        # print(Affine_matrix)
        # input()
        Logr, S = polar_dec(Affine_matrix)
        # print(Logr)
        deformation_gradient.add(vi, Logr, S)

    return deformation_gradient

def Get_DG(Ref_mesh, folder):

    allfiles = natsorted(glob.glob(os.path.join(folder, '*.obj')), alg=ns.IGNORECASE)
    leaf_node = np.loadtxt(os.path.join(folder, '..', 'leaf_id.txt')).astype('int32')
    LOGR = [];S=[];All_V = []
    ref_mesh = pymesh.load_mesh(Ref_mesh)
    Ref_V = ref_mesh.vertices
    Ref_F = ref_mesh.faces
    logr_ind = np.expand_dims(np.expand_dims(np.array([1,2,5], dtype=np.int32), axis=0), axis = 0)
    s_ind = np.expand_dims(np.expand_dims(np.array([0,1,2,4,5,8], dtype=np.int32), axis = 0), axis = 0)

    for infile in allfiles:

        if "-1.obj" in infile:
            continue
        # print(infile)

        mesh = pymesh.load_mesh(infile)
        New_V = mesh.vertices
        dg00 = get_deforma_gradient(Ref_V, New_V, Ref_F)
        LOGR.append(np.expand_dims(dg00.LOGR, axis = 0))
        S.append(np.expand_dims(dg00.S, axis = 0))
        All_V.append(np.expand_dims(New_V, axis = 0))

    LOGR = np.concatenate(LOGR, axis = 0)
    S = np.concatenate(S, axis = 0)
    All_V = np.concatenate(All_V, axis = 0)
    # print(LOGR.shape)
    # print(S.shape)
    LOGR = np.take_along_axis(LOGR, logr_ind, 2)
    S = np.take_along_axis(S, s_ind, 2)
    print(math.isnan(LOGR.sum()))
    print(S.max())
    # import scipy.io as sio
    # sio.savemat('a.mat',{'LOGR': LOGR, 'S':S, 'V':All_V, 'F':Ref_F})
    np.savez(os.path.join(folder, 'geo.npz'), LOGR = LOGR, S = S, partsV = All_V, F = Ref_F, V = Ref_V, leaf_id = leaf_node)


def Get_DG1(Ref_mesh, folder):

    allfiles = natsorted(glob.glob(os.path.join(folder, '*.obj')), alg=ns.IGNORECASE)
    leaf_node = np.loadtxt(os.path.join(folder, '..', 'leaf_id.txt')).astype('int32')
    LOGR = [];S=[];All_V = []
    ref_mesh = pymesh.load_mesh(Ref_mesh)
    Ref_V = ref_mesh.vertices
    Ref_F = ref_mesh.faces
    logr_ind = np.expand_dims(np.expand_dims(np.array([1,2,5], dtype=np.int32), axis=0), axis = 0)
    s_ind = np.expand_dims(np.expand_dims(np.array([0,1,2,4,5,8], dtype=np.int32), axis = 0), axis = 0)

    try:
        os.system('cp %s' % Ref_mesh + ' %s/0.obj' % folder)
    except:
        return
    os.system('./ACAP %s' % folder)
    LOGR = np.loadtxt(os.path.join(folder, 'LOGRNEW.txt')).astype('float32').reshape(-1, 5402, 9)
    S = np.loadtxt(os.path.join(folder, 'S.txt')).astype('float32').reshape(-1, 5402, 9)
    LOGR = LOGR[1:]
    S = S[1:]
    for infile in allfiles:

        if "-1.obj" in infile:
            continue

        mesh = pymesh.load_mesh(infile)
        New_V = mesh.vertices
        All_V.append(np.expand_dims(New_V, axis = 0))

    All_V = np.concatenate(All_V, axis = 0)
    # print(LOGR.shape)
    # print(S.shape)
    LOGR = np.take_along_axis(LOGR, logr_ind, 2)
    S = np.take_along_axis(S, s_ind, 2)
    # print(LOGR.max())
    # print(S.max())
    os.remove(os.path.join(folder, 'S.txt'))
    os.remove(os.path.join(folder, 'LOGRNEW.txt'))
    os.remove(os.path.join(folder, '0.obj'))
    os.remove(os.path.join(folder, '_list.txt'))
    # import scipy.io as sio
    # sio.savemat('a.mat',{'LOGR': LOGR, 'S':S, 'V':All_V, 'F':Ref_F})
    np.savez(os.path.join(folder, 'geo.npz'), LOGR = LOGR, S = S, partsV = All_V, F = Ref_F, V = Ref_V, leaf_id = leaf_node)


def Get_DG2(Ref_mesh, folder):

    allfiles = natsorted(glob.glob(os.path.join(folder, '*.obj')), alg=ns.IGNORECASE)
    leaf_node = np.loadtxt(os.path.join(folder, '..', 'leaf_id.txt')).astype('int32')
    logr_ind = np.expand_dims(np.expand_dims(np.array([1,2,5], dtype=np.int32), axis=0), axis = 0)
    s_ind = np.expand_dims(np.expand_dims(np.array([0,1,2,4,5,8], dtype=np.int32), axis = 0), axis = 0)
    ALL_LOGR = [];ALL_S=[];All_V = [];ALL_ref_V = []
    tmpfolder = os.path.join(folder, 'tmp')

    for infile in allfiles:

        if "-1.obj" in infile:
            continue
        if not os.path.exists(tmpfolder):
            os.makedirs(tmpfolder)
        Ref_mesh = infile.replace('reg/', 'box1/transformed_cube_').replace('_reg', '')
        ref_mesh = pymesh.load_mesh(Ref_mesh)
        Ref_V = ref_mesh.vertices
        Ref_F = ref_mesh.faces
        try:
            os.system('cp %s' % Ref_mesh + ' %s/0.obj' % tmpfolder)
            os.system('cp %s' % infile + ' %s/1.obj' % tmpfolder)
            copyfile(Ref_mesh, os.path.join(tmpfolder, '0.obj'))
            copyfile(infile, os.path.join(tmpfolder, '1.obj'))
        except:
            return
        os.system('./ACAP %s' % tmpfolder)
        LOGR = np.loadtxt(os.path.join(tmpfolder, 'LOGRNEW.txt')).astype('float32').reshape(-1, 5402, 9)
        S = np.loadtxt(os.path.join(tmpfolder, 'S.txt')).astype('float32').reshape(-1, 5402, 9)
        LOGR = LOGR[1:]
        S = S[1:]
        shutil.rmtree(tmpfolder)

        mesh = pymesh.load_mesh(infile)
        New_V = mesh.vertices
        All_V.append(np.expand_dims(New_V, axis = 0))
        ALL_LOGR.append(LOGR)
        ALL_S.append(S)
        ALL_ref_V.append(np.expand_dims(Ref_V, axis = 0))

    All_V = np.concatenate(All_V, axis = 0)
    ALL_ref_V = np.concatenate(ALL_ref_V, axis = 0)
    ALL_LOGR = np.concatenate(ALL_LOGR, axis = 0)
    ALL_S = np.concatenate(ALL_S, axis = 0)
    # print(ALL_LOGR.shape)
    # print(ALL_S.shape)
    LOGR = np.take_along_axis(ALL_LOGR, logr_ind, 2)
    S = np.take_along_axis(ALL_S, s_ind, 2)
    # print(LOGR.max())
    # print(S[leaf_node].max())

    # import scipy.io as sio
    # sio.savemat('a.mat',{'LOGR': LOGR, 'S':S, 'V':All_V, 'F':Ref_F})
    np.savez(os.path.join(folder, 'geonew.npz'), LOGR = LOGR, S = S, partsV = All_V, F = Ref_F, V = ALL_ref_V, leaf_id = leaf_node)

def get_box_quat1(box):
    box = box.squeeze()
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
    return np.expand_dims(box_quat, axis=0)

def Get_box(node_json_file):

    with open(node_json_file, 'r') as fin:
        root_json = json.load(fin)

    box_all = []
    box_quat_all = []

    def traverse(node_json):
        part_id = node_json['id']
        cur_node = {'id': part_id}

        if 'children' in node_json:
            cur_node['children'] = []; cid = 0

            all_child_obj_list = []
            all_child_v = []; all_child_f = []; all_vid = 0
            child_pid2cid = dict()
            for child_json in node_json['children']:
                child_node = traverse(child_json)

                # cur_node['children'].append(child_node)
                # child_pid2cid[child_json['id']] = cid
                # cid += 1

                # all_child_obj_list += child_obj_list
                # all_child_v.append(child_v)
                # all_child_f.append(child_f + all_vid)
                # all_vid += child_v.shape[0]

            # all_v = np.vstack(all_child_v)
            # all_f = np.vstack(all_child_f)
            # cur_node['box'] = mesh_to_obb(all_v, all_f).tolist()
            # cur_node['box'] = mesh_to_obb_and_save(all_v, all_f, part_id).tolist()
            box = np.expand_dims(np.array(node_json['box']).squeeze(), axis=0)
            box_quat = get_box_quat1(box)
            box_all.append((part_id, box))
            box_quat_all.append((part_id, box_quat))
            
            # if len(cur_node['children']) == 0:
                # assert False, 'ERROR: part %d has no child!' % part_id

        else:
            # all_v, all_f = load_obj_list(node_json['objs'])
            box = np.expand_dims(np.array(node_json['box']).squeeze(), axis=0)
            box_quat = get_box_quat1(box)
            box_all.append((part_id, box))
            box_quat_all.append((part_id, box_quat))
        # cur_node['id'] = part_id
        return 1
    root_node = traverse(root_json)
    box_np = np.zeros((len(box_all), 12))
    for i in range(len(box_all)):
        idx, box = box_all[i]
        box_np[idx] = box.squeeze()
    box_quat_np = np.zeros((len(box_quat_all), 10))
    for i in range(len(box_quat_all)):
        idx, box_quat = box_quat_all[i]
        box_quat_np[idx] = box_quat.squeeze()
    return box_np, box_quat_np

def Batch_get_box(folder_id):
    jsonfile = '../data/partnetdata/%s_dhier/%s.json' % (str(cate1), str(folder_id))
    npzfile = '../data/partnetdata/%s_dgeo/%s.npz' % (str(cate1), str(folder_id))
    # savefile = '../data/partnetdata/%s_dgeo/%s.npz' % (str(cate1), str(folder_id))
    try:
        geo_data = np.load(npzfile)
        box_np, box_quat_np = Get_box(jsonfile)
        np.savez(npzfile, LOGR = geo_data['LOGR'], S = geo_data['S'], partsV = geo_data['partsV'], \
            F = geo_data['F'], V = geo_data['V'], leaf_id = geo_data['leaf_id'], box = box_np, box_quat = box_quat_np)
    except:
        print("===============================" + str(folder_id) + "===============================")

def Batch_Get_DG(folder_id):
    # python detect_all_edges.py ${lambdaq}
	# python prepare_partnetobb_dataset.py ${lambdaq}
	# bash gen_html_view.sh ${lambdaq}
    # print(folder_id)
    Ref_mesh = 'ref_cube.obj'
    # folder = '../../../data/structurenet_hire/%s/%s/reg' % (str(cate1), str(folder_id))
    folder = '../../../data/structurenet_hire/%s/%s/box3' % (str(cate1), str(folder_id))
    if 1:#not os.path.exists(os.path.join(folder, 'geo.npz')):
        try:
            Get_DG1(Ref_mesh, folder)
        except:
            print("===============================" + str(folder_id) + "===============================")

    return 1

def collect_data(category, file1):
    form_folder = '../../../data/structurenet_hire/%s' % category
    to_folder1 = '../data/partnetdata/%s_dgeo' % category
    to_folder2 = '../data/partnetdata/%s_dhier' % category
    if not os.path.exists(to_folder1):
        os.makedirs(to_folder1)
    if not os.path.exists(to_folder2):
        os.makedirs(to_folder2)
    #All_shape_id = natsorted(glob.glob(os.path.join(form_folder,'*')), alg=ns.IGNORECASE)
    All_shape_id = np.loadtxt(file1).astype('int32')
    for shape_id in All_shape_id:
        print(shape_id)
        geofile = os.path.join(form_folder, str(shape_id), 'box3', 'geo.npz')
        structfile = os.path.join(form_folder, str(shape_id), 'result_after_merging.json')
        print(geofile)
        print(structfile)
        os.system('cp ' + geofile + ' ' + os.path.join(to_folder1, str(shape_id).split('/')[-1] + '.npz'))
        os.system('cp ' + structfile + ' ' + os.path.join(to_folder2, str(shape_id).split('/')[-1] + '.json'))

def check(folder_id):
    out_dir = '../../../data/structurenet_hire/' + str(cate1)
    regdir = os.path.join(out_dir, str(folder_id),'reg','*.obj')
    allfiles = natsorted(glob.glob(regdir), alg=ns.IGNORECASE)

    for file1 in allfiles:
        try:
            v, f = utils.load_obj(file1)
            # print(file1)
            if len(v) != 5402 or len(f) != 10800:
                print(file1)
                return 1
                # ID.append(i)
                # break
        except:
            print("===============================" + str(file1) + "===============================")
# np.savetxt('error.txt', np.array(ID, dtype= np.int32), delimiter=' ')

def main(file1):
    import multiprocessing
    folder_id = np.loadtxt(file1).astype('int32')

    pool = multiprocessing.Pool(processes=32)
    pool.map(check, folder_id)
    pool.close()
    pool.join()

def main1(file1):
    import multiprocessing
    folder_id = np.loadtxt(file1).astype('int32')

    pool = multiprocessing.Pool(processes=32)
    pool.map(Batch_get_box, folder_id)
    pool.close()
    pool.join()


# main(file1)
main1(file1)
# Batch_Get_DG(1309)
# Batch_get_box(692)
# main('val.txt')
# collect_data(cate1, file1)
# collect_data('Chair', 'test1.txt')
# collect_data(file1)
# Ref_mesh = 'ref_cube.obj'
# folder = '../../../data/structurenet_hire/Chair/40806/reg'
# Get_DG1(Ref_mesh, folder)

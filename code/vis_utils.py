"""
    This file contains utility functions for jupyter notebook visualization.
    Please use jupyter notebook to open vis_box.ipynb or vis_pc.ipynb for results visualization.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
from utils import load_obj
import os
from datav1 import PartGraphShapesDataset

from rand_cmap import rand_cmap
cmap = rand_cmap(300, type='bright', first_color_black=True, last_color_black=False, verbose=False)

def load_semantic_colors(filename):
    semantic_colors = {}
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            semantic, r, g, b = l.rstrip().split()
            semantic_colors[semantic] = (int(r), int(g), int(b))
    return semantic_colors

def draw_box(ax, p, color, rot=None):
    center = p[0: 3]
    lengths = p[3: 6]
    dir_1 = p[6: 9]
    dir_2 = p[9:]

    if rot is not None:
        center = (rot * center.reshape(-1, 1)).reshape(-1)
        dir_1 = (rot * dir_1.reshape(-1, 1)).reshape(-1)
        dir_2 = (rot * dir_2.reshape(-1, 1)).reshape(-1)

    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)

def draw_geo(ax, p, color, rot=None):
    if rot is not None:
        p = (rot * p.transpose()).transpose()

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[color], marker='.')

def draw_geo_with_face(ax, p, f, color, rot=None):
    if rot is not None:
        p = (rot * p.transpose()).transpose()
    # print(p.shape)
    for id, f in enumerate(f):
        verts = [
            p[f[0]].tolist()[0],
            p[f[1]].tolist()[0],
            p[f[2]].tolist()[0]]
        # print(verts)

        tri = a3.art3d.Poly3DCollection([verts])
        tri.set_color(color)
        # tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    # ax.scatter(p[:, 0], p[:, 1], p[:, 2])
    # ax.plot_surface(p[:, 0], p[:, 1], p[:, 2], facecolors=[color])

def draw_edge(ax, e, p_from, p_to, rot=None):
    if rot is not None:
        center1 = (rot * p_from.reshape(-1, 1)).reshape(-1)[0]
        center2 = (rot * p_to.reshape(-1, 1)).reshape(-1)[0]

    edge_type_colors = {
        'ADJ': (1, 0, 0),
        'ROT_SYM': (1, 1, 0),
        'TRANS_SYM': (1, 0, 1),
        'REF_SYM': (0, 0, 0)}

    edge_type_linewidth = {
        'ADJ': 8,
        'ROT_SYM': 6,
        'TRANS_SYM': 4,
        'REF_SYM': 2}

    ax.plot(
        [center1[0, 0], center2[0, 0]], [center1[0, 1], center2[0, 1]], [center1[0, 2], center2[0, 2]],
        c=edge_type_colors[e['type']],
        linestyle=':',
        linewidth=edge_type_linewidth[e['type']])

def output_obj(jsonfile, sem_colors_filename, mtl_file, Tree, PartGraphShapesDataset):
    # Tree.load_category_info('Chair')
    orig_obj = PartGraphShapesDataset.load_object(jsonfile, Tree)
    filepath, fullflname = os.path.split(jsonfile)
    fname, ext = os.path.splitext(fullflname)
    save_file_name = os.path.join(filepath, fname + '.obj')
    Cube_v, Cube_f = load_obj('ref_cube.obj')
    obj_fid = open(save_file_name, 'w')
    obj_fid.write('mtllib color.mtl\n')
    part_boxes, part_geos, edges, part_ids, part_sems = orig_obj.graph(leafs_only=True)
    # sem_colors = load_semantic_colors(filename=sem_colors_filename)
    # for sem in sem_colors:
    #     sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)
    for jj in range(len(part_boxes)):
        if part_geos[jj] is not None:
            v = part_geos[jj].cpu().numpy().reshape(-1, 3)
            f = Cube_f + jj * len(Cube_v)
            for i in range(v.shape[0]):
                obj_fid.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
            obj_fid.write('g part_%s\n' % part_sems[jj].replace('/', '-'))
            obj_fid.write('usemtl m_%s\n' % part_sems[jj].replace('/', '-'))
            for i in range(f.shape[0]):
                obj_fid.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))
    obj_fid.close()
    os.system('cp '+ '%s' % mtl_file +' %s' % os.path.join(filepath, 'color.mtl') + '\n')

def output_obj2pc(jsonfile, Tree):
    # Tree.load_category_info('Chair')
    orig_obj = PartGraphShapesDataset.load_object(jsonfile, Tree)
    filepath, fullflname = os.path.split(jsonfile)
    fname, ext = os.path.splitext(fullflname)
    save_file_name = os.path.join(filepath, fname + '.obj')
    # Cube_v, Cube_f = load_obj('ref_cube.obj')
    obj_fid = open(save_file_name, 'w')
    # obj_fid.write('mtllib color.mtl\n')
    part_boxes, part_geos, edges, part_ids, part_sems = orig_obj.graph(leafs_only=True)
    # sem_colors = load_semantic_colors(filename=sem_colors_filename)
    # for sem in sem_colors:
    #     sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)
    for jj in range(len(part_boxes)):

        v = part_geos[jj].cpu().numpy().reshape(-1, 3)
        # f = Cube_f + jj * len(Cube_v)
        for i in range(v.shape[0]):
            obj_fid.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        # obj_fid.write('g part_%s\n' % part_sems[jj].replace('/', '-'))
        # obj_fid.write('usemtl m_%s\n' % part_sems[jj].replace('/', '-'))
        # for i in range(f.shape[0]):
        #     obj_fid.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))
    obj_fid.close()
    # os.system('cp '+ '%s' % mtl_file +' %s' % os.path.join(filepath, 'color.mtl') + '\n')

def draw_partnet_objects(objects, object_names=None, figsize=None, rep='boxes', \
        leafs_only=False, use_id_as_color=False, visu_edges=True, sem_colors_filename=None, save_file_name = None, mtl_file = None):

    if figsize is not None:
        fig = plt.figure(0, figsize=figsize)
    else:
        fig = plt.figure(0)

    if save_file_name is not None:
        output_obj = True
        Cube_v, Cube_f = load_obj('ref_cube.obj')
        # Cube_v = CUBE['vertices']; Cube_f = CUBE['faces']
    else:
        output_obj = False
    extent = 0.7
    for i, obj in enumerate(objects):
        # load sem colors if provided
        if sem_colors_filename is not None:
            sem_colors = load_semantic_colors(filename=sem_colors_filename[i])
            for sem in sem_colors:
                sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)
        else:
            sem_colors = None

        if output_obj:
            obj_file_name = save_file_name[i]#os.path.join()
            file_path, _ = os.path.split(obj_file_name)
            obj_fid = open(obj_file_name, 'w')
            obj_fid.write('mtllib color.mtl\n')

        part_boxes, part_geos, edges, part_ids, part_sems = obj.graph(leafs_only=leafs_only)

        ax = fig.add_subplot(1, len(objects), i+1, projection='3d')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(extent, -extent)
        ax.set_zlim(-extent, extent)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        # ax.set_aspect('equal')
        ax.set_proj_type('persp')

        if object_names is not None:
            ax.set_title(object_names[i])

        # transform coordinates so z is up (from y up)
        coord_rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        edge_type_order = {
            'ADJ': 1,
            'ROT_SYM': 2,
            'TRANS_SYM': 3,
            'REF_SYM': 4}

        # sort edges by type, so first ones get drawn first
        edges = sorted(edges, key=lambda edge: edge_type_order[edge['type']])

        for jj in range(len(part_boxes)):
            if sem_colors is not None:
                color = sem_colors[part_sems[jj]]
            else:
                color_id = part_ids[jj]
                if use_id_as_color:
                    color_id = jj
                color = cmap(color_id)
            if output_obj:
                v = part_geos[jj].cpu().numpy().reshape(-1, 3)
                f = Cube_f + jj * len(Cube_v)
                for i in range(v.shape[0]):
                    obj_fid.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
                obj_fid.write('g part_%s\n' % part_sems[jj].replace('/', '-'))
                obj_fid.write('usemtl m_%s\n' % part_sems[jj].replace('/', '-'))
                for i in range(f.shape[0]):
                    obj_fid.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

            if rep == 'boxes':
                if part_boxes[jj] is not None:
                    draw_box(
                        ax=ax, p=part_boxes[jj].cpu().numpy().reshape(-1),
                        color=color, rot=coord_rot)
            elif rep == 'geos':
                if part_geos[jj] is not None:
                    draw_geo(
                        ax=ax, p=part_geos[jj].cpu().numpy().reshape(-1, 3),
                        color=color, rot=coord_rot)
            else:
                raise ValueError(f'Unknown representation: {rep}.')

        if output_obj:
            obj_fid.close()
            # os.system('cp '+ '%s' % mtl_file +' %s' % os.path.join(file_path, 'color.mtl') + '\n')

        if visu_edges:
            for jj in range(len(edges)):
                pi_from = edges[jj]['part_a']
                pi_to = edges[jj]['part_b']
                if rep == 'boxes':
                    if part_boxes[pi_from] is not None and part_boxes[pi_to] is not None:
                        p_from = part_boxes[pi_from].view(-1)[:3]
                        p_to = part_boxes[pi_to].view(-1)[:3]
                    else:
                        p_from = None
                        p_to = None
                elif rep == 'geos':
                    if part_geos[pi_from] is not None and part_geos[pi_to] is not None:
                        p_from = part_geos[pi_from].mean(dim=1)
                        p_to = part_geos[pi_to].mean(dim=1)
                    else:
                        p_from = None
                        p_to = None
                else:
                    raise ValueError(f'Unknown representation: {rep}.')

                if p_from is not None and p_to is not None:
                    draw_edge(
                        ax=ax, e=edges[jj],
                        p_from=p_from.cpu().numpy().reshape(-1),
                        p_to=p_to.cpu().numpy().reshape(-1),
                        rot=coord_rot)
                
    plt.tight_layout()
    plt.show()

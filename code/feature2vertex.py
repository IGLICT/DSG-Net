import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import pymesh
import os
import h5py
import torch
import torch.nn as nn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

class Feature2Vertex_pytorch(nn.Module):
    # dg feature to vertex of pytorch version
    def __init__(self, config, gpu = 0):
        super(Feature2Vertex_pytorch, self).__init__()

        self.device = gpu

        self.vertex_dim = 9
        self.resultmax = 0.95
        self.resultmin = -0.95
        self.pointnum = config.num_point
        self.nb = config.nb.to(self.device)
        self.reconmatrix = config.reconmatrix.to(self.device)
        self.weight_vdiff = config.vdiff.to(self.device)
        self.ref_F = config.ref_F

        # self.device = config.device


    def ftoT(self, input_feature):
        resultmax = self.resultmax
        resultmin = self.resultmin
        # input_feature = input_feature.to(self.device)
        batch_size = input_feature.size(0)
        input_feature = input_feature.to(self.device)
        logr = input_feature[:,:,0:3]

        theta = torch.sqrt(torch.sum(logr*logr, dim = 2))

        logr = torch.cat((logr,-logr, torch.zeros((batch_size, self.pointnum, 1), dtype = torch.float32).to(self.device)), 2)

        logr33 = torch.gather(logr, 2, torch.tensor([6,0,1,3,6,2,4,5,6], dtype=torch.int64).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.pointnum, 1).to(self.device))
        logr33 = logr33.view(-1, self.pointnum, 3, 3).contiguous()

        R = torch.eye(3, dtype=torch.float32).repeat(batch_size, self.pointnum, 1, 1).to(self.device)

        condition = torch.eq(theta, 0)

        theta = torch.where(condition, torch.ones_like(theta).to(self.device), theta)

        x = logr33 / torch.unsqueeze(torch.unsqueeze(theta, 2),3)

        sintheta = torch.unsqueeze(torch.unsqueeze(torch.sin(theta), 2), 3)
        costheta = torch.unsqueeze(torch.unsqueeze(torch.cos(theta), 2), 3)

        R_ = R + x*sintheta + torch.matmul(x, x)*(1-costheta)

        condition = condition.unsqueeze(2).repeat(1, 1, 9).view(-1, self.pointnum, 3, 3).contiguous()

        R = torch.where(condition, R, R_)
        S = torch.gather(input_feature, 2, torch.tensor([3,4,5,4,6,7,5,7,8], dtype=torch.int64).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.pointnum, 1).to(self.device)).view(-1, self.pointnum, 3, 3)

        T = torch.matmul(R, S)

        return T

    def Ttov(self, input_feature):
        padding_feature = torch.zeros((input_feature.size(0), 1, 3, 3), dtype=torch.float32).to(self.device)

        padded_input = torch.cat((padding_feature, input_feature), 1)

        nb_T = padded_input[:, self.nb]

        sum_T = nb_T + input_feature.unsqueeze(2)

        bdiff = torch.einsum('abcde,bce->abcd', sum_T, self.weight_vdiff)
        # print(bdiff.shape)
        b = torch.sum(bdiff, 2)
        # print(b.shape)
        v = torch.einsum('ab,cbe->cae', self.reconmatrix, b)

        return v

    def get_vertex_from_feature(self, geofeature):
        device = geofeature.device
        geofeature = geofeature.cpu()
        vertex = self.Ttov(self.ftoT(geofeature))
        vertex = vertex.to(device)
        # print(vertex.shape)
        return vertex

    def alignallmodels(self, deforma_v, id = 0, one = True, modelid = 0):
        model_num = np.shape(deforma_v)[0]
        align_deforma_v = np.zeros_like(deforma_v).astype('float32')
        if not one:
            for idx in range(model_num):
                source = self.all_vertex[idx, id,:,:]
                target = deforma_v[idx]
                T,R,t = best_fit_transform(target, source)
                C = np.ones((np.shape(source)[0], 4))
                C[:,0:3] = target
                align_deforma_v[idx] = (np.dot(T, C.T).T)[:,0:3]
        else:
            source = self.all_vertex[modelid, id,:,:]
            for idx in range(model_num):
                target = deforma_v[idx]
                T,R,t = best_fit_transform(target, source)
                C = np.ones((np.shape(source)[0], 4))
                C[:,0:3] = target
                align_deforma_v[idx] = (np.dot(T, C.T).T)[:,0:3]

        return align_deforma_v

    def save2files(self, geofeature, path, align = False):
        assert(len(path)==len(geofeature))
        vertex = self.Ttov(self.ftoT(geofeature)).cpu().detach().numpy()
        if self.ref_F.min() == 1:
            self.ref_F -= 1
        if align:
            vertex = self.alignallmodels(vertex, id=id, one=one)

        for i in range(len(vertex)):
            new_mesh = pymesh.form_mesh(vertex[i], self.ref_F)
            pymesh.save_mesh(path[i], new_mesh, ascii=True)


class Feature2Vertex():

    def __init__(self, meshinfo, name):
        # self.nb = meshinfo.neighbour
        with tf.device('/cpu:0'):
            self.mesh_Face = meshinfo.Face
            self.all_vertex = meshinfo.all_vertex
            self.pointnum = meshinfo.neighbour.shape[0]
            self.control_idx = meshinfo.control_idx
            self.vertex_dim = 9
            self.resultmax = 0.95
            self.resultmin = -0.95
            self.maxdegree = meshinfo.neighbour.shape[1]

            self.nb = tf.constant(meshinfo.neighbour, dtype = 'int32', shape=[self.pointnum, self.maxdegree], name='nb_relation')
            self.reconmatrix = tf.constant(meshinfo.reconmatrix, dtype = 'float32', shape = [self.pointnum, self.pointnum], name = 'reconmatrix')
            self.weight_vdiff = tf.constant(meshinfo.vdiff, dtype = 'float32', shape = [self.pointnum, self.maxdegree, 3], name = 'wvdiff')
            self.deform_reconmatrix = tf.constant(meshinfo.deform_reconmatrix, dtype = 'float32', shape = [self.pointnum, self.pointnum + len(self.control_idx)], name = 'deform_reconmatrix')

            self.feature2point_pre(name)

    def get_vertex(self, all_feature, path, one = False, id = 0):
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                controlpoint = np.tile([0,0,0], (np.shape(all_feature)[0], 1, 1))
                deforma_v1, deforma_v_align = sess.run([self.deform_vertex, self.deform_vertex_align], feed_dict={self.feature2point: all_feature, self.controlpoint: controlpoint})
                if not one:
                    deforma_v1 = self.alignallmodels(deforma_v1, id=id, one=one)

                self.v2objfile(deforma_v1, path)

        return deforma_v1, deforma_v_align

    def feature2point_pre(self, name = 'f2v_net'):

        with tf.variable_scope(name) as scope:

            self.feature2point = tf.placeholder(tf.float32, [None, self.pointnum, self.vertex_dim], name = 'input_feature2point')

            self.controlpoint = tf.placeholder(tf.float32, [None, len(self.control_idx), 3], name = 'controlpoint')

            self.deform_vertex = self.Ttov(self.ftoT(self.feature2point))
            self.deform_vertex_align = self.deformTtov(self.ftoT(self.feature2point))

    def ftoT(self, input_feature):
        resultmax = self.resultmax
        resultmin = self.resultmin
        batch_size = tf.shape(input_feature)[0]

        logr = input_feature[:,:,0:3]

        theta = tf.sqrt(tf.reduce_sum(logr*logr, axis = 2))

        logr = tf.concat([logr,-logr, tf.zeros([batch_size, self.pointnum, 1], dtype = tf.float32)], 2)

        logr33 = tf.gather(logr, [6,0,1,3,6,2,4,5,6], axis = 2)
        logr33 = tf.reshape(logr33, (-1, self.pointnum, 3, 3))

        R = tf.eye(3, batch_shape=[batch_size, self.pointnum], dtype=tf.float32)

        condition = tf.equal(theta, 0)

        theta = tf.where(condition, tf.ones_like(theta), theta)

        x = logr33 / tf.expand_dims(tf.expand_dims(theta, 2),3)

        sintheta = tf.expand_dims(tf.expand_dims(tf.sin(theta), 2),3)
        costheta = tf.expand_dims(tf.expand_dims(tf.cos(theta), 2),3)

        R_ = R + x*sintheta + tf.matmul(x, x)*(1-costheta)

        condition = tf.reshape(tf.tile(condition, [1, 9]), (-1, self.pointnum, 3, 3))

        R = tf.where(condition, R, R_)
        S = tf.gather(input_feature, [3,4,5,4,6,7,5,7,8], axis = 2)
        S = tf.reshape(S, (-1, self.pointnum, 3, 3))# + tf.eye(3, batch_shape=[batch_size, self.pointnum], dtype=tf.float32)

        T = tf.matmul(R, S)

        return T

    def Ttov(self, input_feature):
        padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, 3, 3], tf.float32)

        padded_input = tf.concat([padding_feature, input_feature], 1)

        nb_T = tf.gather(padded_input, self.nb, axis = 1)

        sum_T = nb_T + tf.expand_dims(input_feature, 2)

        bdiff = tf.einsum('abcde,bce->abcd',sum_T, self.weight_vdiff)

        b = tf.reduce_sum(bdiff, axis = 2)

        v = tf.einsum('ab,cbe->cae', self.reconmatrix, b)

        return v

    def deformTtov(self, input_feature):
        padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, 3, 3], tf.float32)

        padded_input = tf.concat([padding_feature, input_feature], 1)

        nb_T = tf.gather(padded_input, self.nb, axis = 1)

        sum_T = nb_T + tf.expand_dims(input_feature, 2)

        bdiff = tf.einsum('abcde,bce->abcd',sum_T, self.weight_vdiff)

        b = tf.concat([tf.reduce_sum(bdiff, axis = 2), self.controlpoint], axis=1)

        v = tf.einsum('ab,cbe->cae', self.deform_reconmatrix, b)

        return v

    def alignallmodels(self, deforma_v, id = 0, one = True, modelid = 0):
        model_num = np.shape(deforma_v)[0]
        align_deforma_v = np.zeros_like(deforma_v).astype('float32')
        if not one:
            for idx in range(model_num):
                source = self.all_vertex[idx, id,:,:]
                target = deforma_v[idx]
                T,R,t = best_fit_transform(target, source)
                C = np.ones((np.shape(source)[0], 4))
                C[:,0:3] = target
                align_deforma_v[idx] = (np.dot(T, C.T).T)[:,0:3]
        else:
            source = self.all_vertex[modelid, id,:,:]
            for idx in range(model_num):
                target = deforma_v[idx]
                T,R,t = best_fit_transform(target, source)
                C = np.ones((np.shape(source)[0], 4))
                C[:,0:3] = target
                align_deforma_v[idx] = (np.dot(T, C.T).T)[:,0:3]

        return align_deforma_v

    def v2objfile(self, deforma_v, path):

        def savemesh_pymesh(objpath, newv):
            # write and read meshes
            new_mesh = pymesh.form_mesh(newv, self.mesh_Face)
            pymesh.save_mesh(objpath, new_mesh, ascii=True)
        # num = np.shape(deforma_v)[0]
        if not os.path.isdir(os.path.split(path[0])[0]):
            os.makedirs(os.path.split(path[0])[0])
        for i in range(len(deforma_v)):
            # savemesh(self.mesh, path + '/'+ '_'.join(namelist)+ '.obj', deforma_v[i])
            savemesh_pymesh(path[i], deforma_v[i])

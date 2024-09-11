#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, cuda=0, idx=None, xyz=False, first_layer=False, centroid=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if xyz:
            idx = knn(x[:,:3,:], k=k)  # (batch_size, num_points, k)
        elif first_layer :
            idx = knn(centroid[:,:3,:], k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:'+str(cuda)) # 'cuda'

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k, xyz=True) # only using xyz feature find knn
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
    
    
class SGNet(nn.Module):
    def __init__(self, k=5, use_att_fusion=False):
        super(SGNet, self).__init__()
        self.k = k
        self.use_att_fusion = use_att_fusion
        
        self.bn1_s = nn.BatchNorm2d(64)
        self.bn2_s = nn.BatchNorm2d(64)
        self.bn3_s = nn.BatchNorm2d(128)
        # self.bn1_b = nn.BatchNorm2d(64)
        # self.bn2_b = nn.BatchNorm2d(64)
        # self.bn3_b = nn.BatchNorm2d(128)
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(64)
        self.bn3_c = nn.BatchNorm2d(128)
        self.bn_aggr = nn.BatchNorm1d(256)
        self.bn_aggr_s = nn.BatchNorm1d(128)
        # self.bn_aggr_b = nn.BatchNorm1d(128)
        self.bn_aggr_c = nn.BatchNorm1d(128)

        self.conv1_semantic = nn.Sequential(nn.Conv2d(10*2, 64, kernel_size=1, bias=False),
                                   self.bn1_s,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_semantic = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2_s,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_semantic = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3_s,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # self.conv1_box = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
        #                            self.bn1_b,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2_box = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
        #                            self.bn2_b,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv3_box = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
        #                            self.bn3_b,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
        self.conv1_centroid = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_centroid = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_centroid = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv_aggr = nn.Sequential(nn.Conv1d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn_aggr,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.conv_aggr_s = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn_aggr_s,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # self.conv_aggr_b = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            self.bn_aggr_b,
        #                            nn.LeakyReLU(negative_slope=0.2))
        
        self.conv_aggr_c = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn_aggr_c,
                                   nn.LeakyReLU(negative_slope=0.2))
        if self.use_att_fusion:
            self.fusion = MultiModalFusion(modal_num=3, with_weight=1)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(0,2,1)
        semantic = x[:,:10,:] #x[:,:528,:]
        # box = x[:,10:13,:] #x[:,528:531,:]
        centroid = x[:,13:16,:]#x[:,531:534,:]
        num_node = x.shape[2]
        if num_node < 5:
            k = num_node
        else:
            k = 5
        semantic = get_graph_feature(semantic, k, xyz=False, first_layer=True, centroid=centroid) # only using xyz feature find knn
        semantic = self.conv1_semantic(semantic)
        semantic1 = semantic.max(dim=-1, keepdim=False)[0]

        semantic = get_graph_feature(semantic1, k)
        semantic = self.conv2_semantic(semantic)
        semantic2 = semantic.max(dim=-1, keepdim=False)[0]

        semantic = get_graph_feature(semantic2, k)
        semantic = self.conv3_semantic(semantic)
        semantic3 = semantic.max(dim=-1, keepdim=False)[0]
        
        semantic_final = self.conv_aggr_s(torch.cat((semantic1, semantic2, semantic3), dim=1))
        
        # box = get_graph_feature(box, k, xyz=False,  first_layer=True, centroid=centroid) # only using xyz feature find knn
        # box = self.conv1_box(box)
        # box1 = box.max(dim=-1, keepdim=False)[0]

        # box = get_graph_feature(box1, k)
        # box = self.conv2_box(box)
        # box2 = box.max(dim=-1, keepdim=False)[0]

        # box = get_graph_feature(box2, k)
        # box = self.conv3_box(box)
        # box3 = box.max(dim=-1, keepdim=False)[0]
        
        # box_final = self.conv_aggr_b(torch.cat((box1, box2, box3), dim=1))
        
        centroid = get_graph_feature(centroid, k, xyz=True) # only using xyz feature find knn
        centroid = self.conv1_centroid(centroid)
        centroid1 = centroid.max(dim=-1, keepdim=False)[0]

        centroid = get_graph_feature(centroid1, k)
        centroid = self.conv2_centroid(centroid)
        centroid2 = centroid.max(dim=-1, keepdim=False)[0]

        centroid = get_graph_feature(centroid2, k)
        centroid = self.conv3_centroid(centroid)
        centroid3 = centroid.max(dim=-1, keepdim=False)[0]
        
        centroid_final = self.conv_aggr_c(torch.cat((centroid1, centroid2, centroid3), dim=1))

        if self.use_att_fusion:
            x = self.fusion([semantic_final, centroid_final])
        else:
            x = torch.cat((semantic_final, centroid_final), dim=1)

        x = self.conv_aggr(x)
        
        x = x.permute(0, 2, 1)  # [node_num, 32]
        x = x.squeeze(0)  # [node_num, 32]
        return x
    
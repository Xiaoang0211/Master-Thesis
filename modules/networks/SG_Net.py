import time
import torch
import numpy as np
from tqdm import tqdm, trange
# from torch_geometric.nn import GCNConv
from modules.networks.neural_tensor_networks import PoolingWithAttention, TensorNetworkModule, GeM
from utilities.sg_pr_utils import *
from tensorboardX import SummaryWriter
# from warmup_scheduler import GradualWarmupScheduler
import os
import modules.EncodeNetworks.dgcnn as dgcnn
import torch.nn as nn
from collections import OrderedDict
from sklearn import metrics

from modules.networks.node_transformer import *


class SG(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, feat_dim_in):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SG, self).__init__()
        self.args = args
        self.feat_dim_in = feat_dim_in
        self.feat_dim_out = self.args.filters_3
        self.num_head = self.args.num_head
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.gem_pooling = GeM()
        # self.node_transformer = StackedSelfAttention(feature_dim=self.feat_dim_out, num_heads=self.num_head, num_layers=2)
        # self.node_transformer = TransformerDecoder(self.feat_dim_out, self.num_head, num_layers=3)
        self.node_transformer1 = StackedCrossAttention(feature_dim=self.feat_dim_out, num_heads=self.num_head, num_layers=2)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        bias_bool = False # TODO
        self.dgcnn_gat_conv1 = nn.Sequential(
            nn.Conv2d(self.feat_dim_in*2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_point_conv1 = nn.Sequential(
            nn.Conv2d(self.feat_dim_in * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_gat_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1*2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_point_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_gat_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_point_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.args.filters_3*2,
                                                      self.args.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.args.filters_3), nn.LeakyReLU(negative_slope=0.2))
        
        
        
    def edge_conv_pass(self, x):
        self.k = self.args.K
        object = x[:,:100,:] # Bx3xN
        structure = x[:,100:,:]   # BxfxN
        
        object = dgcnn.get_graph_feature(object, k=self.k, cuda=self.args.cuda)  # Bx2fxNxk
        object = self.dgcnn_point_conv1(object)
        object1 = object.max(dim=-1, keepdim=False)[0]
        object = dgcnn.get_graph_feature(object1, k=self.k, cuda=self.args.cuda)
        object = self.dgcnn_point_conv2(object)
        object2 = object.max(dim=-1, keepdim=False)[0]
        object = dgcnn.get_graph_feature(object2, k=self.k, cuda=self.args.cuda)
        object = self.dgcnn_point_conv3(object)
        object3 = object.max(dim=-1, keepdim=False)[0]

        structure = dgcnn.get_graph_feature(structure, k=self.k, cuda=self.args.cuda)    #Bx6xNxk
        structure = self.dgcnn_gat_conv1(structure)
        structure1 = structure.max(dim=-1, keepdim=False)[0]
        structure = dgcnn.get_graph_feature(structure1, k=self.k, cuda=self.args.cuda)
        structure = self.dgcnn_gat_conv2(structure)
        structure2 = structure.max(dim=-1, keepdim=False)[0]
        structure = dgcnn.get_graph_feature(structure2, k=self.k, cuda=self.args.cuda)
        structure = self.dgcnn_gat_conv3(structure)
        structure3 = structure.max(dim=-1, keepdim=False)[0]
        


        x = torch.cat((object3, structure3), dim=1)
        x = self.dgcnn_conv_end(x)

        x = x.permute(0, 2, 1)  # [node_num, 32]
        return x
    
    def forward(self, src_features, ref_features):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        
        src_features = src_features.permute(0,2,1)
        ref_features = ref_features.permute(0,2,1)
                
        # permuted_cross_features_1 = src_features.permute(0,2,1)
        # permuted_cross_features_2 = ref_features.permute(0,2,1)
        
        # get node embeddings with lower dimension
        abstract_features_1 = self.edge_conv_pass(src_features) # node_num x feature_size(filters-3)
        abstract_features_2 = self.edge_conv_pass(ref_features)  #BXNXF
        
        # # self attention
        # self_features_1 = self.node_self_attention_1(abstract_features_1)
        # self_features_2= self.node_self_attention_2(abstract_features_2)
        
        # section 1
        # cross attention
        cross_features_1, cross_features_2 = self.node_transformer1(abstract_features_1, abstract_features_2)
        
        # GeM pooling
        pooled_features_1 = self.gem_pooling(cross_features_1)
        pooled_features_2 = self.gem_pooling(cross_features_2)

        # # optional: node level embedding according to SimNN

        # compute similarity scores
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = scores.permute(0,2,1) # bx1xf
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)
        
        # # compute cosine similarities
        # pooled_features_1 = pooled_features_1.squeeze(-1)  # Shape: (num_pairs, feature_dim)
        # pooled_features_2 = pooled_features_2.squeeze(-1)  # Shape: (num_pairs, feature_dim)
        # # Cosine similarity is computed along the feature_dim axis
        # score = torch.nn.functional.cosine_similarity(pooled_features_1, pooled_features_2, dim=1)
        return score, src_features.permute(0,2,1), ref_features.permute(0,2,1)
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 100),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        # x shape is expected to be [660, 200]
        return self.layers(x)
    
class MLP_200(nn.Module):
    def __init__(self):
        super(MLP_200, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(100, 512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        # )
        # self.layers = nn.Sequential(
        #     nn.Linear(200, 512),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(512),  # Normalize after the first layer
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(256),  # Normalize after the second layer
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(128),  # Normalize after the third layer
        # )
        # self.layers = nn.Sequential(
        #     nn.Linear(200, 512),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(512),  # Normalize after the first layer
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(256),  # Normalize after the second layer
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(128),  # Normalize after the third layer
        # )
    
    def forward(self, x):
        # x shape is expected to be [660, 200]
        return self.layers(x)
    
class SG_TransformerDecoder(torch.nn.Module):
    def __init__(self, args, feat_dim_in, feat_dim_out, num_head):
        super(SG_TransformerDecoder, self).__init__()
        self.args = args
        self.feat_dim_in = feat_dim_in
        self.feat_dim_out = feat_dim_out
        self.num_head = num_head
        
        self.setup_layers()
        
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons
        
    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.MLP = nn.Sequential(
            nn.Linear(200, 512),
            nn.LeakyReLU(),
            nn.LayerNorm(512),  # Normalize after the first layer
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),  # Normalize after the second layer
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),  # Normalize after the third layer
        )
        # self.MLP = nn.Sequential(
        #     nn.Linear(200, 512),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        # )
        self.gem_pooling = GeM()
        self.node_transformer = TransformerDecoder(self.feat_dim_out, self.num_head, num_layers=2)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        bias_bool = False # TODO
        
        self.dgcnn_point_conv1 = nn.Sequential(
            nn.Conv2d(self.feat_dim_in * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_point_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_point_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.args.filters_3,
                                                      self.args.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.args.filters_3), nn.LeakyReLU(negative_slope=0.2))
        
        
    def edge_conv_pass(self, x):
        # x shape: (22, 200, 30)
        # what we want: (22, 128, 30)
        self.k = self.args.K
        object = x.permute(0,2,1).reshape(-1,200) # Bx3xN
        object = self.MLP(object)
        object = object.reshape(-1,30,128).permute(0,2,1)
        
        
        object = dgcnn.get_graph_feature(object, k=self.k, cuda=self.args.cuda)  # Bx2fxNxk
        object = self.dgcnn_point_conv1(object)
        object1 = object.max(dim=-1, keepdim=False)[0]
        object = dgcnn.get_graph_feature(object1, k=self.k, cuda=self.args.cuda)
        object = self.dgcnn_point_conv2(object)
        object2 = object.max(dim=-1, keepdim=False)[0]
        object = dgcnn.get_graph_feature(object2, k=self.k, cuda=self.args.cuda)
        object = self.dgcnn_point_conv3(object)
        object3 = object.max(dim=-1, keepdim=False)[0]
  
        x = self.dgcnn_conv_end(object3)
        
        x = x.permute(0, 2, 1)  # [node_num, 32]
        return x
    
    def forward(self, src_features, ref_features):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        src_features = src_features.permute(0,2,1)
        ref_features = ref_features.permute(0,2,1)
        
        # get node embeddings with lower dimension
        abstract_features_1 = self.edge_conv_pass(src_features) # node_num x feature_size(filters-3)
        abstract_features_2 = self.edge_conv_pass(ref_features)  #BXNXF
        
        concat_features = torch.cat((abstract_features_1, abstract_features_2), dim=1)
        
        # section 1
        # cross attention
        cross_features_1, cross_features_2 = self.node_transformer(concat_features)
        # cross_features_1, cross_features_2 = self.node_transformer1(abstract_features_1, abstract_features_2)
        
        # GeM pooling
        pooled_features_1 = self.gem_pooling(cross_features_1)
        pooled_features_2 = self.gem_pooling(cross_features_2)

        # # optional: node level embedding according to SimNN

        # compute similarity scores
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = scores.permute(0,2,1) # bx1xf
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)
        return score, src_features.permute(0,2,1), ref_features.permute(0,2,1)
    
class SG_one_channel(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """

    def __init__(self, args, feat_dim_in):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SG_one_channel, self).__init__()
        self.args = args
        self.feat_dim_in = feat_dim_in
        self.feat_dim_out = self.args.filters_3
        self.num_head = self.args.num_head
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.gem_pooling = GeM()
        # self.node_transformer = StackedSelfAttention(feature_dim=self.feat_dim_out, num_heads=self.num_head, num_layers=2)
        # self.node_transformer = TransformerDecoder(self.feat_dim_out, self.num_head, num_layers=3)
        self.node_transformer1 = StackedCrossAttention(feature_dim=self.feat_dim_out, num_heads=self.num_head, num_layers=2)
        self.tensor_network = TensorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        self.mlp = MLP_200()
        bias_bool = False # TODO
        self.dgcnn_gat_conv1 = nn.Sequential(
            nn.Conv2d(self.feat_dim_in*2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_point_conv1 = nn.Sequential(
            nn.Conv2d(self.feat_dim_in * 2, self.args.filters_1, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_1),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_gat_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1*2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_point_conv2 = nn.Sequential(
            nn.Conv2d(self.args.filters_1 * 2, self.args.filters_2, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_2),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_gat_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_point_conv3 = nn.Sequential(
            nn.Conv2d(self.args.filters_2*2, self.args.filters_3, kernel_size=1, bias=bias_bool),
            nn.BatchNorm2d(self.args.filters_3),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.args.filters_3,
                                                      self.args.filters_3, kernel_size=1, bias=bias_bool),
                                            nn.BatchNorm1d(self.args.filters_3), nn.LeakyReLU(negative_slope=0.2))
        
        
    def edge_conv_pass(self, x):
        # x shape: (22, 200, 30)
        # what we want: (22, 128, 30)
        self.k = self.args.K
        object = x.permute(0,2,1).reshape(-1,200) # Bx3xN
        object = self.mlp(object)
        object = object.reshape(-1,30,128).permute(0,2,1)
        # x = object
        
        object = dgcnn.get_graph_feature(object, k=self.k, cuda=self.args.cuda)  # Bx2fxNxk
        object = self.dgcnn_point_conv1(object)
        object1 = object.max(dim=-1, keepdim=False)[0]
        object = dgcnn.get_graph_feature(object1, k=self.k, cuda=self.args.cuda)
        object = self.dgcnn_point_conv2(object)
        object2 = object.max(dim=-1, keepdim=False)[0]
        object = dgcnn.get_graph_feature(object2, k=self.k, cuda=self.args.cuda)
        object = self.dgcnn_point_conv3(object)
        object3 = object.max(dim=-1, keepdim=False)[0]
  
        x = self.dgcnn_conv_end(object3)

        x = x.permute(0, 2, 1)  # [node_num, 32]
        return x
    
    def forward(self, src_features, ref_features, all_features, features_count):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        
        src_scenes, ref_scenes = self.reshape_features(all_features, features_count)
        
        src_features = src_features.permute(0,2,1)
        ref_features = ref_features.permute(0,2,1)
                
        # permuted_cross_features_1 = src_features.permute(0,2,1)
        # permuted_cross_features_2 = ref_features.permute(0,2,1)
        
        # get node embeddings with lower dimension
        abstract_features_1 = self.edge_conv_pass(src_features) # node_num x feature_size(filters-3)
        abstract_features_2 = self.edge_conv_pass(ref_features)  #BXNXF
        
        
        # # self attention
        # self_features_1 = self.node_self_attention_1(abstract_features_1)
        # self_features_2= self.node_self_attention_2(abstract_features_2)
        
        # section 1
        # cross attention
        cross_features_1, cross_features_2 = self.node_transformer1(abstract_features_1, abstract_features_2)
        
        # GeM pooling
        pooled_features_1 = self.gem_pooling(cross_features_1)
        pooled_features_2 = self.gem_pooling(cross_features_2)

        # # optional: node level embedding according to SimNN

        # compute similarity scores
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = scores.permute(0,2,1) # bx1xf
        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)
        
        # # compute cosine similarities
        # pooled_features_1 = pooled_features_1.squeeze(-1)  # Shape: (num_pairs, feature_dim)
        # pooled_features_2 = pooled_features_2.squeeze(-1)  # Shape: (num_pairs, feature_dim)
        # # Cosine similarity is computed along the feature_dim axis
        # score = torch.nn.functional.cosine_similarity(pooled_features_1, pooled_features_2, dim=1)
        return score, src_scenes, ref_scenes
    
    
    def reshape_features(self, features, src_ref_counts):
        """
        Reshape and divide a feature array into src and ref scene arrays.

        :param features: A torch tensor of shape [num_objects, feature_dim]
        :param num_objects_per_pair: A torch tensor indicating the number of objects in each scene pair
        :param src_ref_counts: A torch tensor indicating the number of objects in src and ref scenes for each pair
        :return: Two torch tensors for src and ref scenes, each of shape (num_scenes, num_objects, feature_dim)
        """
        src_scenes = []
        ref_scenes = []
        start_idx = 0

        for src_count, ref_count in src_ref_counts:
            # Indices for src and ref scenes
            end_idx = start_idx + src_count + ref_count

            # Slice the feature array for the current scene pair
            scene_pair_features = features[start_idx:end_idx]

            # Separate src and ref features and reshape
            src_scene_features = scene_pair_features[:src_count].view(1, src_count, -1)
            ref_scene_features = scene_pair_features[src_count:].view(1, ref_count, -1)

            # Append to lists
            src_scenes.append(src_scene_features)
            ref_scenes.append(ref_scene_features)

            # Update start index for next iteration
            start_idx = end_idx

        return src_scenes, ref_scenes
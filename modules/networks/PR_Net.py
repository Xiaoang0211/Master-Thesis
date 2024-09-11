import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from modules.networks.node_embed import MultiModalEncoder # module for node level embedding
from modules.networks.SG_Net import SG, SG_one_channel, SG_TransformerDecoder # module for graph level embedding
import time

class PR_Net(nn.Module):
    def __init__(self, cfg):
        super(PR_Net, self).__init__()
        # Model Specific params for node encoder
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim
        self.modal_emb_dim = cfg.model.modal_emb_dim
        self.freeze_sgaligner = cfg.train.freeze_sgaligner
        
        self.args_sg = cfg.args_sg

        # Networks
        self.node_encoder = MultiModalEncoder(modules = self.modules, rel_dim = self.rel_dim, attr_dim=self.attr_dim)
        # self.graph_encoder = SG(self.args_sg, feat_dim_in=self.modal_emb_dim)
        self.graph_encoder = SG_one_channel(self.args_sg, feat_dim_in=self.modal_emb_dim)
        # self.graph_encoder = SG_TransformerDecoder(self.args_sg, feat_dim_in=self.modal_emb_dim, feat_dim_out=128, num_head=4)
        
        self.pretrained_pointnet = '/app/git_repos/sgaligner_w_SE3/output/Scan3R/sgaligner/point_SE3_wo_rescans/snapshots/epoch-50.pth.tar'
        self.pretrained_sgnet = '/app/git_repos/sgaligner_w_SE3/output/Scan3R/sgaligner/sgnet_SE3_w_rescans_sc/snapshots/epoch-50.pth.tar'
        
        if "point" in self.modules:
            self.load_pretrained_sgaligner(self.pretrained_pointnet, type='object')
        if "sgnet" in self.modules:
            self.load_pretrained_sgaligner(self.pretrained_sgnet, type='structure')

        # eventually freeze the sgaligner
        if self.freeze_sgaligner:
            for param in self.node_encoder.parameters():
                param.requires_grad = False
        
                
    def forward(self, data_dict):
        output_dict, embs_meta = self.node_encoder(data_dict)
        # reshape node embeddings
        src_features, ref_features = self.reshape_features(output_dict['joint'], 
                                                           embs_meta['graph_per_obj_count'])
        # time2 = time.time()
        scores, src_emb, ref_emb = self.graph_encoder(src_features, ref_features, output_dict['joint'], embs_meta['graph_per_obj_count']) # similarity scores
        # time3 = time.time()

        # print(time3 -time2)
        output_dict['src_emb'] = src_emb
        output_dict['ref_emb'] = ref_emb
        output_dict['ref_id'] = data_dict['scene_ids'][:,1]
        output_dict['graph_per_obj_count'] = embs_meta['graph_per_obj_count']
        output_dict['predictions'] = scores
        output_dict['targets'] = embs_meta['target']
        return output_dict # , (time3 - time2, src_features.shape[0])
    
    
    def load_pretrained_sgaligner(self, model_path, type='structure'):
        self.pretrained = model_path
        if type == 'structure':
            checkpoint = torch.load(self.pretrained)
            pretrained_weights = checkpoint['model']
            
            structure_encoder_keys = {k.replace('structure_encoder.', ''): v for k, v in pretrained_weights.items() if k.startswith('structure_encoder')}
            structure_embedding_keys = {k.replace('structure_embedding.', ''): v for k, v in pretrained_weights.items() if k.startswith('structure_embedding')}

            # Update the node_encoder weights
            self.node_encoder.structure_encoder.load_state_dict(structure_encoder_keys, strict=True)
            self.node_encoder.structure_embedding.load_state_dict(structure_embedding_keys, strict=True)

        elif type == 'object':
            checkpoint = torch.load(self.pretrained)
            pretrained_weights = checkpoint['model']
            
            object_encoder_keys = {k.replace('object_encoder.', ''): v for k, v in pretrained_weights.items() if k.startswith('object_encoder')}
            object_embedding_keys = {k.replace('object_embedding.', ''): v for k, v in pretrained_weights.items() if k.startswith('object_embedding')}

            # Update the node_encoder weights
            self.node_encoder.object_encoder.load_state_dict(object_encoder_keys, strict=True)
            self.node_encoder.object_embedding.load_state_dict(object_embedding_keys, strict=True)
        
    
    
    
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
   
        src_features_padded = [self.zero_padding_and_sampling(src_scene) for src_scene in src_scenes]
        ref_features_padded = [self.zero_padding_and_sampling(ref_scene) for ref_scene in ref_scenes]
        
        # Concatenate the lists of scenes to form the final tensors
        src_features = torch.cat(src_features_padded, dim=0)
        ref_features = torch.cat(ref_features_padded, dim=0)

        return src_features, ref_features
    
    def zero_padding_and_sampling(self, scene):
        """
        Transferring the data to torch and creating a hash table with the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        node_num = scene.shape[1]
        joint_emb_dim = scene.shape[2]
        device = scene.device
        
        if node_num > self.args_sg.node_num:
            sampled_index = np.random.choice(node_num, self.args_sg.node_num, replace=False)
            sampled_index.sort()
            scene = scene[:,sampled_index,:]
        elif node_num < self.args_sg.node_num:
            padding_size = self.args_sg.node_num - node_num

            # Create a zero tensor for padding
            padding = torch.zeros(1, padding_size, joint_emb_dim).to(device)

            # Concatenate the original tensor with the padding along dimension 1
            scene = torch.cat((scene, padding), dim=1)
        
        return scene
        



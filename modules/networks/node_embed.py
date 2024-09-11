import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.EncodeNetworks.pointnet import PointNetfeat
from modules.EncodeNetworks.dgcnn import SGNet

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=128, out_dim=32, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.l2(x)
        return x

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

class MultiModalEncoder(nn.Module):
    def __init__(self, modules, rel_dim, attr_dim, hidden_units=[3, 128, 128], heads = [2, 2], emb_dim = 100, pt_out_dim = 256,
                       dropout = 0.0, attn_dropout = 0.0, instance_norm = False):
        super(MultiModalEncoder, self).__init__()
        self.modules = modules
        self.pt_out_dim = pt_out_dim
        self.rel_dim = rel_dim
        self.emb_dim = emb_dim
        self.attr_dim =  attr_dim
        self.hidden_units = hidden_units
        self.heads = heads
        
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.instance_norm = instance_norm
        self.inner_view_num = len(self.modules) # Point Net + Structure Encoder + Meta Encoder
        
        # object encoder
        if 'point' in self.modules:
            self.object_encoder = PointNetfeat(global_feat=True, batch_norm=True, point_size=3, input_transform=False, feature_transform=False, out_size=self.pt_out_dim)
        else:
            # raise NotImplementedError
            pass
        self.object_embedding = nn.Linear(self.pt_out_dim, self.emb_dim)
        
        # structure encoder        
        self.structure_encoder = SGNet()
        self.structure_embedding = nn.Linear(256, self.emb_dim)
        
        # self.structure_embedding = ProjectionHead() # if linear: nn.Linear(256, self.emb_dim)
        
        # self.fusion = MultiModalFusion(modal_num=self.inner_view_num, with_weight=1)
        
    def forward(self, data_dict):
        tot_object_points = data_dict['tot_obj_pts'].permute(0, 2, 1)
        batch_size = len(data_dict['overlap'])
        
        
        embs = {}
        embs_meta = {}
       
        for module in self.modules:     
            if module == "sgnet":
                structure_embed = None
                src_node_start_idx = 0
                ref_node_start_idx = 0
                for idx in range(batch_size):
                    src_node_count = data_dict['graph_per_obj_count'][idx][0]
                    ref_node_count = data_dict['graph_per_obj_count'][idx][1]
                    src_node_end_idx = src_node_start_idx + src_node_count
                    ref_node_end_idx = ref_node_start_idx + ref_node_count
                    src_node_features = data_dict['src_node_feat'][src_node_start_idx:src_node_end_idx]
                    ref_node_features = data_dict['ref_node_feat'][ref_node_start_idx:ref_node_end_idx]
                    
                    src_structure_embedding = self.structure_encoder(src_node_features)
                    ref_structure_embedding = self.structure_encoder(ref_node_features)
                    
                    structure_embed = torch.cat([src_structure_embedding, ref_structure_embedding]) if structure_embed is None else \
                                    torch.cat([structure_embed, src_structure_embedding, ref_structure_embedding]) 
                emb = self.structure_embedding(structure_embed)
            elif module in ['point', 'dgcnn']:
                emb = self.object_encoder(tot_object_points)
                emb = self.object_embedding(emb) 
            else:
                # raise NotImplementedError
                pass
            
            embs[module] = emb
        
        if len(self.modules) > 1:
            all_embs = []
            for module in self.modules:
                all_embs.append(embs[module])
            
            # joint_emb = self.fusion(all_embs) # attention based fusion
            joint_emb = torch.cat(all_embs, dim=1) # simple concatenation
            embs['joint'] = joint_emb
            
        embs_meta['graph_per_obj_count'] = data_dict['graph_per_obj_count']
        embs_meta['target'] = data_dict['target']
        return embs, embs_meta
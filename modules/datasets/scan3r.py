import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder

import sys
sys.path.append('..')

from utilities.utils_node_match import common, scan3r
from utilities.sg_pr_utils import load_txt_list
from utilities.point_cloud import *

class Scan3RDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.use_predicted = cfg.use_predicted
        self.pc_resolution = cfg.val.pc_res if split == 'val' or 'test' else cfg.train.pc_res
        self.anchor_type_name = cfg.preprocess.anchor_type_name
        self.model_name = cfg.model_name
        self.scan_type = cfg.scan_type
        self.lower_bound = cfg.lower_bound
        self.upper_bound = cfg.upper_bound
        self.w_or_wo_rescans = cfg.w_or_wo_rescans
        self.num_pos = cfg.train.num_pos
        self.num_neg = cfg.train.num_neg
        self.portion_size = 100
        self.rand_mod_rate = cfg.data.rand_mod_rate
        
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname

        self.data_root_dir = cfg.data.root_dir
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.meta_dir = f"meta_{self.lower_bound}_{self.upper_bound}_{self.w_or_wo_rescans}_rescans"
        
        
        self.mode = 'orig' if self.split == 'train' else cfg.val.data_mode

        self.anchor_data_filename = osp.join(self.scans_files_dir, '{}/{}/anchors{}_{}.json'.format(self.mode, self.meta_dir, self.anchor_type_name, split))
        if self.split == 'test':
            self.anchor_data_all_pos_filename = osp.join(self.scans_files_dir, '{}/{}/anchors{}_{}_all_pos.json'.format(self.mode, self.meta_dir, self.anchor_type_name, split))
            self.anchor_data_all_pos = common.load_json(self.anchor_data_all_pos_filename)[:]
            
        self.all_subscan_ids_filename = osp.join(self.scans_files_dir, '{}/{}/{}_scans_subscenes.txt'.format(self.mode, self.meta_dir, self.split))
        print('[INFO] Reading from {} with point cloud resolution - {}'.format(self.anchor_data_filename, self.pc_resolution))
        self.anchor_data = common.load_json(self.anchor_data_filename)[:]
        self.all_subscan_ids = load_txt_list(self.all_subscan_ids_filename)
                
        if split == 'val' and cfg.val.overlap_low != cfg.val.overlap_high:
            final_anchor_data = []
            for anchor_data_idx in self.anchor_data:
                if anchor_data_idx['overlap'] >= cfg.val.overlap_low and anchor_data_idx['overlap'] < cfg.val.overlap_high:
                    final_anchor_data.append(anchor_data_idx)
            
            self.anchor_data = final_anchor_data
        
        self.is_training = self.split == 'train'
        self.do_augmentation = False if self.split == 'val' or 'test' else cfg.train.use_augmentation

        self.rot_factor = cfg.train.rot_factor
        self.augment_noise = cfg.train.augmentation_noise

        # Jitter
        self.scale = 0.01
        self.clip = 0.05

        # Random Rigid Transformation
        self._rot_mag = 45.0 # grad
        self._trans_mag = 0.5 # meter
        
    def __len__(self):
        return len(self.all_subscan_ids)
    
    def _modify_decimal_label(self, label, p, max_change=50):
        if random.random() < p:
            # Randomly decide to add or subtract
            change = random.randint(-max_change, max_change)
            # Apply the change and ensure the label stays within bounds
            new_label = max(0, min(500, label + change))
            return new_label
        else:
            return label

    def _modify_array_of_decimal_labels(self, labels, p, max_change=50):
        return np.array([self._modify_decimal_label(label, p, max_change) for label in labels])
    
    def _sample_pairs(self, query_subscan_id):
        if self.split == "train" or self.split == "val":
            # Initial direct positives finding
            positives_ref = [pair_dict["ref"] for pair_dict in self.anchor_data if pair_dict["src"] == query_subscan_id]
            positives_src = [pair_dict["src"] for pair_dict in self.anchor_data if pair_dict["ref"] == query_subscan_id]
            initial_positives = set(positives_ref + positives_src)
            
            # For positive sampling, only use the initial direct positives
            positive_samples = random.sample(list(initial_positives), min(self.num_pos, len(initial_positives)))
            
            # Now, find all positives, including indirect, to ensure they're not considered in negative samples
            all_positives = set(initial_positives)
            new_positives = set(initial_positives)
            
            while new_positives:
                current_new_positives = set()
                for pos in new_positives:
                    # Find positives of each positive
                    additional_positives_ref = [pair_dict["ref"] for pair_dict in self.anchor_data if pair_dict["src"] == pos and pair_dict["ref"] not in all_positives]
                    additional_positives_src = [pair_dict["src"] for pair_dict in self.anchor_data if pair_dict["ref"] == pos and pair_dict["src"] not in all_positives]
                    
                    # Update the current new positives
                    current_new_positives.update(additional_positives_ref + additional_positives_src)
                
                # Update the master list of all positives with the new findings
                all_positives.update(current_new_positives)
                
                # Prepare for the next iteration
                new_positives = current_new_positives

            # Ensure the query ID is not considered for either positives or negatives
            all_positives.discard(query_subscan_id)
            
            # Prepare the set of all subscan IDs, excluding the query and all positives for negatives sampling
            training_ids_set = set(self.all_subscan_ids)
            negatives_candidates = training_ids_set - all_positives
            
            # Random sample 3 negatives from the updated set of candidates
            negatives_samples = random.sample(list(negatives_candidates), min(self.num_neg, len(negatives_candidates)))
            
            return positive_samples, negatives_samples
        
        elif self.split == 'test':
            # Extract all scan pairs contained in the list anchor_train.json
            positives_ref = [pair_dict["ref"] for pair_dict in self.anchor_data_all_pos if pair_dict["src"] == query_subscan_id]
            positives_src = [pair_dict["src"] for pair_dict in self.anchor_data_all_pos if pair_dict["ref"] == query_subscan_id]
            initial_positives = set(positives_ref + positives_src)
            
            all_positives = set(initial_positives)
            new_positives = set(initial_positives)
            
            while new_positives:
                current_new_positives = set()
                for pos in new_positives:
                    # Find positives of each positive
                    additional_positives_ref = [pair_dict["ref"] for pair_dict in self.anchor_data if pair_dict["src"] == pos and pair_dict["ref"] not in all_positives]
                    additional_positives_src = [pair_dict["src"] for pair_dict in self.anchor_data if pair_dict["ref"] == pos and pair_dict["src"] not in all_positives]
                    
                    # Update the current new positives
                    current_new_positives.update(additional_positives_ref + additional_positives_src)
                
                # Update the master list of all positives with the new findings
                all_positives.update(current_new_positives)
                
                # Prepare for the next iteration
                new_positives = current_new_positives

            # Ensure the query ID is not considered for either positives or negatives
            all_positives.discard(query_subscan_id)
            
            # # use this when only test for rescans
            # query_id = query_subscan_id[:36]
            # for positive in list(all_positives):
            #     if positive[:36] == query_id:
            #         all_positives.discard(positive)
            
            # # use this when only test for overlap with upper limit
            # for positive in list(initial_positives):
            #     overlap = [pair_dict["overlap"] for pair_dict in self.anchor_data_all_pos if (pair_dict["src"] == query_subscan_id and pair_dict["ref"] == positive) or (pair_dict["src"] == positive and pair_dict["ref"] == query_subscan_id)]
            #     overlap = overlap[0]
            #     if overlap > 0.9:
            #         all_positives.discard(positive)
            
            
            all_negatives = set(self.all_subscan_ids)
            all_negatives -= {query_subscan_id}
            all_negatives -= all_positives
            
            return list(all_positives), list(all_negatives)
        
    def _map_tensor_values(self, tensor, mapping_dict):
        # Use vectorize method for efficient element-wise operation
        vmap = np.vectorize(mapping_dict.get)
        return vmap(tensor)
    
    def _calculate_dir_vec(self, object_points, order_array):
        centroids = np.mean(object_points, axis=1)
        # vectors = np.zeros((len(order_array), 3))
        # for i, (from_idx, to_idx) in enumerate(order_array):
        #     vectors[i] = centroids[from_idx] - centroids[to_idx]
        return centroids
    
    def _get_bounding_boxes(self, batch_point_clouds):
        
        min_coords = np.min(batch_point_clouds, axis=1)  # Shape will be (20, 3)
        max_coords = np.max(batch_point_clouds, axis=1)  # Shape will be (20, 3)

        # Calculate dimensions (length, width, height) for each bounding box
        return max_coords - min_coords  # Shape will be (20, 3)

    
    def _process_pair(self, graph_data):
        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        overlap = graph_data['overlap'] if 'overlap' in graph_data else -1.0
        
        # Centering
        src_points = scan3r.load_plydata_npy(osp.join(self.scans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None)
        ref_points = scan3r.load_plydata_npy(osp.join(self.scans_scenes_dir, '{}/data.npy'.format(ref_scan_id)), obj_ids = None)
        # if positive pair, then both are in the same coordinate
        # we firstly add the rotation and translation to the src_points and then 
        # center both of then with their own pcl_center
        
        if self.split == 'train':
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(src_points, axis=0)
            else:
                pcl_center = np.mean(ref_points, axis=0)
        else:
            pcl_center = np.mean(src_points, axis=0)

        src_data_dict = common.load_pkl_data(osp.join(self.scans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(self.scans_files_dir, '{}/data/{}.pkl'.format(self.mode, ref_scan_id)))
        
        src_mapping_dict = dict(zip(src_data_dict['objects_id'], src_data_dict['global_objects_id']))
        ref_mapping_dict = dict(zip(ref_data_dict['objects_id'], ref_data_dict['global_objects_id']))
        
        src_semantic_pairs = np.array(src_data_dict['pairs'])
        ref_semantic_pairs = np.array(ref_data_dict['pairs'])
        
        src_semantic_pairs = self._map_tensor_values(src_semantic_pairs, src_mapping_dict)
        ref_semantic_pairs = self._map_tensor_values(ref_semantic_pairs, ref_mapping_dict) 
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']
      
        src_global_objects_id =  self._modify_array_of_decimal_labels(src_data_dict['global_objects_id'], self.rand_mod_rate)
        ref_global_objects_id =  self._modify_array_of_decimal_labels(ref_data_dict['global_objects_id'], self.rand_mod_rate)
                     
        # decimal to binary
        src_global_objects_id = np.array([list(format(num, '010b')) for num in src_global_objects_id]).astype(np.int64)
        ref_global_objects_id = np.array([list(format(num, '010b')) for num in ref_global_objects_id]).astype(np.int64)
        
        anchor_obj_ids = graph_data['anchorIds'] if 'anchorIds' in graph_data else src_object_ids
        global_object_ids = np.concatenate((src_data_dict['objects_cat'], ref_data_dict['objects_cat']))
        
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id != 0]
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id in src_object_ids and anchor_obj_id in ref_object_ids]
        
        # if self.split == 'train':
        #     anchor_cnt = 2 if int(0.3 * len(anchor_obj_ids)) < 1 else int(0.3 * len(anchor_obj_ids))
        #     anchor_obj_ids = anchor_obj_ids[:anchor_cnt]

        src_edges = src_data_dict['edges']
        ref_edges = ref_data_dict['edges']

        src_object_points = src_data_dict['obj_points'][self.pc_resolution] - pcl_center
        ref_object_points = ref_data_dict['obj_points'][self.pc_resolution] - pcl_center
        
        src_object_id2idx = src_data_dict['object_id2idx']
        e1i_idxs = np.array([src_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) # e1i
        e1j_idxs = np.array([src_object_id2idx[object_id] for object_id in src_data_dict['objects_id'] if object_id not in anchor_obj_ids]) # e1j
        
        ref_object_id2idx = ref_data_dict['object_id2idx']
        e2i_idxs = np.array([ref_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) + src_object_points.shape[0] # e2i
        e2j_idxs = np.array([ref_object_id2idx[object_id] for object_id in ref_data_dict['objects_id'] if object_id not in anchor_obj_ids]) + src_object_points.shape[0] # e2j

        src_bounding_box = self._get_bounding_boxes(src_object_points)
        ref_bounding_box = self._get_bounding_boxes(ref_object_points)
        
        
        R = gen_random_rot_deg(roll_range=(-self._rot_mag, self._rot_mag), 
                            pitch_range=(-self._rot_mag, self._rot_mag))
        t = gen_random_trans(x_range=(-self._trans_mag, self._trans_mag), 
                        y_range=(-self._trans_mag, self._trans_mag), 
                        z_range=(-self._trans_mag, self._trans_mag))
        
        ref_obj_transformed = apply_transformation(ref_object_points, R, t)
        
        src_centroids = self._calculate_dir_vec(src_object_points, src_edges)
        ref_centroids = self._calculate_dir_vec(ref_obj_transformed, ref_edges)
        
        src_centroids_expanded = src_centroids[:, np.newaxis, :]
        ref_centroids_expanded = ref_centroids[:, np.newaxis, :]
        
        tot_object_points = torch.cat([torch.from_numpy(src_object_points-src_centroids_expanded), 
                                       torch.from_numpy(ref_obj_transformed-ref_centroids_expanded)]).type(torch.FloatTensor)
        
        # node features
        src_node_features = torch.from_numpy(np.concatenate((src_global_objects_id, src_bounding_box, src_centroids), axis=1))
        ref_node_features = torch.from_numpy(np.concatenate((ref_global_objects_id, ref_bounding_box, ref_centroids), axis=1)) # not transformed
        
      
        data_dict = {} 
        data_dict['obj_ids'] = np.concatenate([src_object_ids, ref_object_ids])
        data_dict['graph_per_obj_count'] = np.array([src_object_points.shape[0], ref_object_points.shape[0]])
        data_dict['tot_obj_pts'] = tot_object_points
        
        data_dict['e1i'] = e1i_idxs
        data_dict['e1i_count'] = e1i_idxs.shape[0]
        data_dict['e2i'] = e2i_idxs
        data_dict['e2i_count'] = e2i_idxs.shape[0]
        data_dict['e1j'] = e1j_idxs
        data_dict['e1j_count'] = e1j_idxs.shape[0]
        data_dict['e2j'] = e2j_idxs
        data_dict['e2j_count'] = e2j_idxs.shape[0]
        data_dict['tot_obj_count'] = data_dict['obj_ids'].shape[0]

        data_dict['global_obj_ids'] = global_object_ids
        data_dict['scene_ids'] = [src_scan_id, ref_scan_id]     
        data_dict['src_node_feat'] = src_node_features.float()
        data_dict['ref_node_feat'] = ref_node_features.float()
        data_dict['overlap'] = overlap
        
        return data_dict
    
    def _generate_pseudo_negative_data(self, query_subscan_id, negative):
        negative_pair = {}
        negative_pair['src'] = query_subscan_id
        negative_pair['ref'] = negative
        negative_pair['overlap'] = 0
        negative_pair['anchorIds'] = []
        return negative_pair
    
    def _generate_pseudo_positive_data(self, query_subscan_id, positive):
        positive_pair = {}
        positive_pair['src'] = query_subscan_id
        positive_pair['ref'] = positive
        positive_pair['overlap'] = 0.5 # pseudo overlap
        positive_pair['anchorIds'] = []
        return positive_pair
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")

    def __getitem__(self, idx): # get one batch
        query_subscan_id = self.all_subscan_ids[idx]
        
        positives, negatives = self._sample_pairs(query_subscan_id)
        if self.split != 'test':
            query_pairs = []
            # positive pairs
            for positive in positives:
                positive_data = [pair_dict for pair_dict in self.anchor_data if (pair_dict["src"] == query_subscan_id and pair_dict["ref"] == positive) or (pair_dict["ref"] == query_subscan_id and pair_dict["src"] == positive)]  
                positive_data = positive_data[0]
                if positive_data and positive_data["ref"] == query_subscan_id:
                    positive_data["src"], positive_data["ref"] = positive_data["ref"], positive_data["src"]
                data_dict_pos = self._process_pair(positive_data)
                query_pairs.append(data_dict_pos)
    
            for negative in negatives:
                negative_data = self._generate_pseudo_negative_data(query_subscan_id, negative)
                data_dict_neg = self._process_pair(negative_data)
                query_pairs.append(data_dict_neg)
            data_dict = self._concatenate_pos_neg_pairs(query_pairs) 
            # data_dict['tot_obj_count_single_scene'] = np.append(data_dict['e1j_count'][self.num_pos], data_dict['tot_obj_count'] - data_dict['e1j_count'][self.num_pos])
            return data_dict
        else:
            pos_and_neg_ids = positives + negatives
            total = len(pos_and_neg_ids)
            portions = [self.portion_size * i for i in range(0, total // self.portion_size + 1)]
            portions.append(total)
            # portions = [0, 100, 200, 300, 400, 500, 600, 700, 800, 840]
            data_dicts = []
            for i in range(len(portions) - 1):
                start_idx = portions[i]
                end_idx = portions[i + 1]
                query_pairs = []
                if i == 0 and len(positives) != 0:
                    positive_portion = positives
                    negative_portion = pos_and_neg_ids[start_idx + len(positive_portion):end_idx]
                    
                    for positive in positive_portion:
                        positive_data = [pair_dict for pair_dict in self.anchor_data_all_pos if (pair_dict["src"] == query_subscan_id and pair_dict["ref"] == positive) or (pair_dict["ref"] == query_subscan_id and pair_dict["src"] == positive)]  
                        if len(positive_data) != 0:
                            positive_data = positive_data[0]
                        else:
                            positive_data = self._generate_pseudo_positive_data(query_subscan_id, positive)
                        if positive_data and positive_data["ref"] == query_subscan_id:
                            positive_data["src"], positive_data["ref"] = positive_data["ref"], positive_data["src"]
                        data_dict_pos = self._process_pair(positive_data)
                        query_pairs.append(data_dict_pos)
            
                    for negative in negative_portion:
                        negative_data = self._generate_pseudo_negative_data(query_subscan_id, negative)
                        data_dict_neg = self._process_pair(negative_data)
                        query_pairs.append(data_dict_neg)    
                else:
                    negative_portion = pos_and_neg_ids[start_idx:end_idx]
                    for negative in negative_portion:
                        negative_data = self._generate_pseudo_negative_data(query_subscan_id, negative)
                        data_dict_neg = self._process_pair(negative_data)
                        query_pairs.append(data_dict_neg)
                data_dict = self._concatenate_pos_neg_pairs(query_pairs)     
                data_dicts.append(data_dict)
            return data_dicts
    
    def _collate_entity_idxs(self, batch, batch_concate=False): 
        e1i = np.concatenate([data['e1i'] for data in batch])
        e2i = np.concatenate([data['e2i'] for data in batch])
        e1j = np.concatenate([data['e1j'] for data in batch])
        e2j = np.concatenate([data['e2j'] for data in batch])
        
        e1i_start_idx = 0 
        e2i_start_idx = 0 
        e1j_start_idx = 0 
        e2j_start_idx = 0 
        prev_obj_cnt = 0
        
        for idx in range(len(batch)):
            if batch_concate: # doubled
                e1i_end_idx = e1i_start_idx + sum(batch[idx]['e1i_count'])
                e2i_end_idx = e2i_start_idx + sum(batch[idx]['e2i_count'])
                e1j_end_idx = e1j_start_idx + sum(batch[idx]['e1j_count'])
                e2j_end_idx = e2j_start_idx + sum(batch[idx]['e2j_count'])

                e1i[e1i_start_idx : e1i_end_idx] += prev_obj_cnt
                e2i[e2i_start_idx : e2i_end_idx] += prev_obj_cnt
                e1j[e1j_start_idx : e1j_end_idx] += prev_obj_cnt
                e2j[e2j_start_idx : e2j_end_idx] += prev_obj_cnt
            
                prev_obj_cnt += sum(batch[idx]['tot_obj_count'])
            else:
                e1i_end_idx = e1i_start_idx + batch[idx]['e1i_count']
                e2i_end_idx = e2i_start_idx + batch[idx]['e2i_count']
                e1j_end_idx = e1j_start_idx + batch[idx]['e1j_count']
                e2j_end_idx = e2j_start_idx + batch[idx]['e2j_count']
                
                e1i[e1i_start_idx : e1i_end_idx] += prev_obj_cnt
                e2i[e2i_start_idx : e2i_end_idx] += prev_obj_cnt
                e1j[e1j_start_idx : e1j_end_idx] += prev_obj_cnt
                e2j[e2j_start_idx : e2j_end_idx] += prev_obj_cnt
            
                prev_obj_cnt += batch[idx]['tot_obj_count']
                
            e1i_start_idx, e2i_start_idx, e1j_start_idx, e2j_start_idx = e1i_end_idx, e2i_end_idx, e1j_end_idx, e2j_end_idx
                
        e1i = e1i.astype(np.int32)
        e2i = e2i.astype(np.int32)
        e1j = e1j.astype(np.int32)
        e2j = e2j.astype(np.int32)

        return e1i, e2i, e1j, e2j


    def _collate_feats(self, batch, key):
        feats = torch.cat([data[key] for data in batch])
        return feats
    
    def _concatenate_pos_neg_pairs(self, query_pairs):
        tot_object_points = self._collate_feats(query_pairs, 'tot_obj_pts')
        
        
        src_node_features = self._collate_feats(query_pairs, 'src_node_feat')
        ref_node_features = self._collate_feats(query_pairs, 'ref_node_feat')
        
        data_dict = {}
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['e1i'], data_dict['e2i'], data_dict['e1j'], data_dict['e2j'] = self._collate_entity_idxs(query_pairs)

        data_dict['e1i_count'] = np.stack([data['e1i_count'] for data in query_pairs])
        data_dict['e2i_count'] = np.stack([data['e2i_count'] for data in query_pairs])
        data_dict['e1j_count'] = np.stack([data['e1j_count'] for data in query_pairs])
        data_dict['e2j_count'] = np.stack([data['e2j_count'] for data in query_pairs])
        data_dict['tot_obj_count'] = np.stack([data['tot_obj_count'] for data in query_pairs])
        data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in query_pairs])
        
        data_dict['graph_per_obj_count'] = np.stack([data['graph_per_obj_count'] for data in query_pairs])
        data_dict['scene_ids'] = np.stack([data['scene_ids'] for data in query_pairs])
        data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in query_pairs])
        # data_dict['pcl_center'] = np.stack([data['pcl_center'] for data in query_pairs])   
        data_dict['src_node_feat'] = src_node_features 
        data_dict['ref_node_feat'] = ref_node_features
        # data_dict['rotation'] = np.stack([data['rotation'] for data in query_pairs])
        # data_dict['translation'] = np.stack([data['translation'] for data in query_pairs])
        data_dict['overlap'] = np.stack([data['overlap'] for data in query_pairs])
        data_dict['target'] = torch.tensor(np.where(data_dict['overlap'] != 0, 1, 0))
        # data_dict['num_pairs_per_query'] = data_dict['overlap'].shape[0]

        return data_dict
    
    # def collate_fn(self, batch):
    #     tot_object_points = self._collate_feats(batch, 'tot_obj_pts')
        
    #     src_node_features = self._collate_feats(batch, 'src_node_feat')
    #     ref_node_features = self._collate_feats(batch, 'ref_node_feat')

    #     data_dict = {}
    #     data_dict['tot_obj_pts'] = tot_object_points
    #     data_dict['e1i'], data_dict['e2i'], data_dict['e1j'], data_dict['e2j'] = self._collate_entity_idxs(batch, batch_concate=True)

    #     data_dict['e1i_count'] = np.concatenate([data['e1i_count'] for data in batch])
    #     data_dict['e2i_count'] = np.concatenate([data['e2i_count'] for data in batch])
    #     data_dict['e1j_count'] = np.concatenate([data['e1j_count'] for data in batch])
    #     data_dict['e2j_count'] = np.concatenate([data['e2j_count'] for data in batch])
    #     data_dict['tot_obj_count'] = np.concatenate([data['tot_obj_count'] for data in batch])
    #     data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in batch])
        
    #     data_dict['graph_per_obj_count'] = np.concatenate([data['graph_per_obj_count'] for data in batch])
    #     # data_dict['graph_per_edge_count'] = np.concatenate([data['graph_per_edge_count'] for data in batch])
    #     data_dict['src_node_feat'] = src_node_features 
    #     data_dict['ref_node_feat'] = ref_node_features
    #     data_dict['scene_ids'] = np.concatenate([data['scene_ids'] for data in batch])
    #     data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in batch])
    #     # data_dict['pcl_center'] = np.concatenate([data['pcl_center'] for data in batch])
        
    #     data_dict['overlap'] = np.concatenate([data['overlap'] for data in batch])
    #     data_dict['target'] = torch.tensor(np.where(data_dict['overlap'] != 0, 1, 0))
    #     data_dict['batch_size'] = len(batch)
        
    #     # for key, value in data_dict.items():
    #     #     if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
    #     #         print(f"Shape of {key}: {value.shape}")
    #     #     else:
    #     #         print(f"Value of {key}: {value}")

    #     return data_dict
    
    def collate_fn(self, batch):
        data_dict_batched = batch[0]
        return data_dict_batched
        
if __name__ == '__main__':
    from configs_node_match import config_scan3r_gt
    cfg = config_scan3r_gt.make_cfg()
    scan3r_ds = Scan3RDataset(cfg, split='val')
    print(len(scan3r_ds))
    scan3r_ds[0]    
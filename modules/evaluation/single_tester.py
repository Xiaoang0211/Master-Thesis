from typing import Dict
import torch
import ipdb
import numpy as np
from tqdm import tqdm
import math
import os.path as osp
import json
import random

import time
from statistics import mean

from modules.evaluation.base_tester import BaseTester
from utilities.utils_node_match import torch_util
from utilities.utils_node_match.common import get_log_string
from utilities.utils_node_match import scan3r
from sklearn.metrics.pairwise import cosine_similarity
# from modules.networks.node_matching import NodeMatching

class SingleTester_base(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)
        # self.NodeMatching = NodeMatching(num_correspondences=3, dual_normalization=False)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)
    
    def correlation_matrix(self, tensor_a, tensor_b):
        tensor_a = tensor_a.squeeze(0)  # Now of shape (8, 128)
        tensor_b = tensor_b.squeeze(0)  # Now of shape (16, 128)
        
        num_nodes_a = tensor_a.shape[0]
        num_nodes_b = tensor_b.shape[0]

        corr_matrix = torch.zeros((num_nodes_a, num_nodes_b))

        for i in range(num_nodes_a):
            for j in range(num_nodes_b):
                # Get the nodes
                node_a = tensor_a[i]
                node_b = tensor_b[j]

                # Calculate correlation
                corr_matrix[i, j] = torch.corrcoef(torch.stack((node_a, node_b)))[0, 1]
        
        return corr_matrix
    
    def cosine_similarity_matrix(self, tensor_a, tensor_b, k=5):
        # k = 5 #min(tensor_a.shape[0], tensor_b.shape[0])
        # Normalize the tensors
        norm_a = torch.nn.functional.normalize(tensor_a, p=2, dim=1)
        norm_b = torch.nn.functional.normalize(tensor_b, p=2, dim=1)

        # Calculate cosine similarity
        # Using matrix multiplication to compute pairwise similarity
        sim_matrix = torch.mm(norm_a, norm_b.transpose(0, 1))
        sim_matrix_flat = sim_matrix.flatten()
        top_values, top_indices_1d = torch.topk(sim_matrix_flat, k)

        # Convert the 1D indices to 2D indices
        rows = top_indices_1d // sim_matrix.size(1)
        cols = top_indices_1d % sim_matrix.size(1)

        # Combine rows and columns for 2D indices
        top_indices_2d = torch.stack((rows, cols), dim=1)
        return top_indices_2d, top_values

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)     
        inner_iteration = 0  
        
        correct_match = 0
        wrong_match = 0
        ambiguity = 0 
        # tuples = []
        for iteration, data_dicts in pbar:
            predictions_all = []
            target_all = []
            src_embs = []
            ref_embs = []
            obj_count = []
            # obj_ids_all = []
            global_ids_all = []
            ref_ids = []
            src_scan_id = data_dicts[0]['scene_ids'][0,0]
            iteration += 1
            for data_dict in data_dicts:  # iterate over each data_dict in the list
                # on start
                inner_iteration += 1      
                data_dict = torch_util.to_cuda(data_dict)

                self.before_test_step(inner_iteration, data_dict)
                # test step
                torch.cuda.synchronize()
                self.timer.add_prepare_time()
                output_dict = self.test_step(inner_iteration, data_dict)  # output_dict contains the node embeddings
                torch.cuda.synchronize()
                self.timer.add_process_time()
                # eval step for node matching
     
                results_dict = self.eval_step(inner_iteration, data_dict, output_dict)
                # append the result predictions target to list
                src_embs.append(output_dict['src_emb'])
                ref_embs.append(output_dict['ref_emb'])
                ref_ids.append(output_dict['ref_id'])
                obj_count.append(output_dict['graph_per_obj_count'])
                predictions_all.append(output_dict['predictions'])
                target_all.append(output_dict['targets'])
                # obj_ids_all.append(data_dict['obj_ids'])
                global_ids_all.append(data_dict['global_obj_ids'])
                # if tuple[1] != 2:
                #     tuples.append(tuple)
                
                message = f'{self.timer.tostring()}'
                pbar.set_description(message)
                pbar.update(1)
                torch.cuda.empty_cache()

            predictions_all = torch_util.release_cuda(predictions_all)
            target_all = torch_util.release_cuda(target_all)    
            
            predictions_all = np.concatenate(predictions_all)
            target_all = np.concatenate(target_all)
            # src_embs = torch.cat(src_embs, dim=0)
            # ref_embs = torch.cat(ref_embs, dim=0)
            ref_ids = np.concatenate(ref_ids)
            obj_count = np.concatenate(obj_count)
            # obj_ids_all = np.concatenate(obj_ids_all)
            global_ids_all = np.concatenate(global_ids_all)
            
            for score_type in self.scores.keys():
                if '%' in score_type:
                    k = int(score_type[:-1])
                    k = math.floor(total_iterations*k*0.01)
                    topk_idx = self.eval_step_graph_match(predictions_all, target_all, k, self.scores[score_type])
                elif score_type == '5':
                    num_positives = np.sum(target_all)
                    top_5_idx = np.argsort(predictions_all)[-5:]
                    top_5_ids = ref_ids[top_5_idx]
                    src_scan_id = src_scan_id.reshape(1,)
                    top_5_ids = np.concatenate((src_scan_id, top_5_ids))
                    if min(top_5_idx) <= (num_positives - 1):
                        print(src_scan_id)
                        print(top_5_idx)
                        print(top_5_ids)
                        print('Hello')
                else: 
                    k = int(score_type)
                    topk_idx = self.eval_step_graph_match(predictions_all, target_all, k, self.scores[score_type])
                    # topk_idx = topk_idx[0]
                    # if k == 1: # get the top 1 embeddings
                    #     src_emb = src_embs[topk_idx]
                    #     ref_emb = ref_embs[topk_idx]
                    #     ref_id_top1 = ref_ids[topk_idx]
                    #     obj_count_top1 = obj_count[topk_idx]
                    #     start_idx = np.sum(obj_count[:topk_idx])
                    #     end_idx_src = start_idx + obj_count[topk_idx][0]
                    #     end_idx_ref = end_idx_src + obj_count[topk_idx][1]
                    
                        
            # adding re-ranking
            
            ## run registration with GeoTransformer
            # get top 1 retrieved points
            # points are the not downsampled version, features are used to determine the node correspondence
            # remove zero paddings
            # src_emb = src_emb[:obj_count_top1[0],:]
            # ref_emb = ref_emb[:obj_count_top1[1],:]
            # src_obj_ids = obj_ids_all[start_idx:end_idx_src]
            # ref_obj_ids = obj_ids_all[end_idx_src:end_idx_ref]
            # src_global_ids = global_ids_all[start_idx:end_idx_src]
            # ref_global_ids = global_ids_all[end_idx_src:end_idx_ref]
            # cross correlation between node features
            
            # src_corr_indices, ref_corr_indices, corr_scores = self.NodeMatching(src_emb, ref_emb)
            # extract the top k node correspondences
            # random sampling 
            # indices, similarities = self.cosine_similarity_matrix(src_emb, ref_emb, self.k)
            # for index in indices:
            #     if src_obj_ids[index[0]] == ref_obj_ids[index[1]] and src_global_ids[index[0]] == ref_global_ids[index[1]]:
            #         correct_match += 1
            #     else:
            #         wrong_match += 1
            #         if src_global_ids[index[0]] == ref_global_ids[index[1]]:
            #             ambiguity += 1
            # run registration on the top k node correspondences
            
        
        for recall_type, recall_list in self.scores.items():
            recall = sum(recall_list)/len(recall_list)
            results_dict['alignment_metrics']['pr_recall'][recall_type] = recall
        
        # results_dict['alignment_metrics']['correct_match_rate'] = correct_match/(self.k*total_iterations)
        # results_dict['alignment_metrics']['ambiguity_rate'] = ambiguity/(self.k*total_iterations)
        
        self.after_test_epoch()
        # array_tuples = np.array(tuples)
        # tuples_0 = array_tuples[:,0]
        # print("Mean time for graph embedding: ", np.mean(tuples_0))
        self.print_metrics(results_dict)
        
class SingleTester_simple(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)        
        for iteration, data_dicts in pbar:
            # on start
            predictions_all = []
            target_all = []
            obj_count = []
            obj_ids_all = []
            all_ref_ids = []
            overlap_all = []
            rank_lists = dict()
            
            for data_dict in data_dicts:  # iterate over each data_dict in the list
                # on start
                iteration += 1      
                data_dict = torch_util.to_cuda(data_dict)

                self.before_test_step(iteration, data_dict)
                # test step
                torch.cuda.synchronize()
                self.timer.add_prepare_time()
                output_dict = self.test_step(iteration, data_dict)  # output_dict contains the node embeddings
                # after test step begins the 
                torch.cuda.synchronize()
                self.timer.add_process_time()

                # eval step for node matching
                results_dict = self.eval_step(iteration, data_dict, output_dict, rank_lists)
                predictions_all.append(output_dict['predictions'])
                target_all.append(output_dict['targets'])
                obj_count.append(output_dict['graph_per_obj_count'])
                obj_ids_all.append(data_dict['obj_ids'])
                all_ref_ids.append(output_dict['ref_id'])
                overlap_all.append(data_dict['overlap'])
                
                # eval step for registration
                message = f'{self.timer.tostring()}'
                pbar.set_description(message)
                torch.cuda.empty_cache()

            predictions_all = torch_util.release_cuda(predictions_all)
            target_all = torch_util.release_cuda(target_all)      
            predictions_all = np.concatenate(predictions_all)
            target_all = np.concatenate(target_all)
            obj_count = np.concatenate(obj_count)
            obj_ids_all = np.concatenate(obj_ids_all)
            all_ref_ids = np.concatenate(all_ref_ids)
            overlap_all = np.concatenate(overlap_all)
             
            topk_idx = int(self.get_topk_idx(predictions_all, k=1))
            start_idx = np.sum(obj_count[:topk_idx])
            end_idx_src = start_idx + obj_count[topk_idx][0]
            end_idx_ref = end_idx_src + obj_count[topk_idx][1]
            src_obj_count = obj_count[topk_idx][0]
            src_obj_ids = obj_ids_all[start_idx:end_idx_src]
            ref_obj_ids = obj_ids_all[end_idx_src:end_idx_ref]
            pair_obj_ids = np.concatenate((src_obj_ids, ref_obj_ids))
            src_scan_id = data_dict['scene_ids'][0][0]
            ref_scan_id = all_ref_ids[topk_idx]
            pcl_center = data_dict['pcl_center'][0]
            overlap = overlap_all[topk_idx]
            # print(overlap)
            
            
            src_ref_pair = src_scan_id + ref_scan_id
            
            if overlap != 0: # only correct match shall be registered
                results_registration = self.eval_step_registration(rank_lists[src_ref_pair], 
                                                                src_obj_count,
                                                                pair_obj_ids,
                                                                src_scan_id,
                                                                ref_scan_id,
                                                                pcl_center)

        self.after_test_epoch()
        # with open('results_dict.json', 'w') as json_file:
        #     json.dump(results_dict, json_file, indent=4)
            
        with open('results_registration.json', 'w') as json_file:
            json.dump(results_registration, json_file, indent=4)
        self.print_metrics(results_dict)
        self.print_metrics(results_registration)
        
        
class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)
        # self.NodeMatching = NodeMatching(num_correspondences=3, dual_normalization=False)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)
    
    def correlation_matrix(self, tensor_a, tensor_b):
        tensor_a = tensor_a.squeeze(0)  # Now of shape (8, 128)
        tensor_b = tensor_b.squeeze(0)  # Now of shape (16, 128)
        
        num_nodes_a = tensor_a.shape[0]
        num_nodes_b = tensor_b.shape[0]

        corr_matrix = torch.zeros((num_nodes_a, num_nodes_b))

        for i in range(num_nodes_a):
            for j in range(num_nodes_b):
                # Get the nodes
                node_a = tensor_a[i]
                node_b = tensor_b[j]

                # Calculate correlation
                corr_matrix[i, j] = torch.corrcoef(torch.stack((node_a, node_b)))[0, 1]
        
        return corr_matrix
    
    def cosine_similarity_matrix(self, tensor_a, tensor_b, k=5):
        #k = 5 #min(tensor_a.shape[0], tensor_b.shape[0])
        # Normalize the tensors
        norm_a = torch.nn.functional.normalize(tensor_a, p=2, dim=1)
        norm_b = torch.nn.functional.normalize(tensor_b, p=2, dim=1)

        # Calculate cosine similarity
        # Using matrix multiplication to compute pairwise similarity
        sim_matrix = torch.mm(norm_a, norm_b.transpose(0, 1))
        sim_matrix_flat = sim_matrix.flatten()
        top_values, top_indices_1d = torch.topk(sim_matrix_flat, k)

        # Convert the 1D indices to 2D indices
        rows = top_indices_1d // sim_matrix.size(1)
        cols = top_indices_1d % sim_matrix.size(1)

        # Combine rows and columns for 2D indices
        top_indices_2d = torch.stack((rows, cols), dim=1)
        return top_indices_2d, top_values

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)     
        inner_iteration = 0  
        delta_ts = []
        
        for iteration, data_dicts in pbar:
            predictions_all = []
            target_all = []
            src_embs = []
            ref_embs = []
            obj_count = []
            obj_ids_all = []
            global_ids_all = []
            ref_ids = []
            src_scan_id = data_dicts[0]['scene_ids'][0,0]
            iteration += 1
            for data_dict in data_dicts:  # iterate over each data_dict in the list
                # on start
                inner_iteration += 1      
                data_dict = torch_util.to_cuda(data_dict)

                self.before_test_step(inner_iteration, data_dict)
                # test step
                torch.cuda.synchronize()
                self.timer.add_prepare_time()
                output_dict = self.test_step(inner_iteration, data_dict)  # output_dict contains the node embeddings
                torch.cuda.synchronize()
                self.timer.add_process_time()

                # eval step for node matching
                results_dict = self.eval_step(inner_iteration, data_dict, output_dict)
                # append the result predictions target to list
                src_embs.extend(output_dict['src_emb'])
                ref_embs.extend(output_dict['ref_emb'])
                ref_ids.append(output_dict['ref_id'])
                obj_count.append(output_dict['graph_per_obj_count'])
                predictions_all.append(output_dict['predictions'])
                target_all.append(output_dict['targets'])
                obj_ids_all.append(data_dict['obj_ids'])
                global_ids_all.append(data_dict['global_obj_ids'])
                
                message = f'{self.timer.tostring()}'
                pbar.set_description(message)
                pbar.update(1)
                torch.cuda.empty_cache()

            predictions_all = torch_util.release_cuda(predictions_all)
            target_all = torch_util.release_cuda(target_all)      
            predictions_all = np.concatenate(predictions_all)
            target_all = np.concatenate(target_all)
            src_embs = torch_util.release_cuda(src_embs)
            ref_embs = torch_util.release_cuda(ref_embs)
            ref_ids = np.concatenate(ref_ids)
            obj_count = np.concatenate(obj_count)
            obj_ids_all = np.concatenate(obj_ids_all)
            global_ids_all = np.concatenate(global_ids_all)
            
            
            
            
            z = 50
            topz_idx = np.argsort(predictions_all)[-z:]
            for score_type in self.scores.keys(): # initial retrieval score_type = 1, 2, 5, 1%, 5%
                t1 = time.time()
                if '%' in score_type:
                    recall_topk = int(score_type[:-1])
                    recall_topk = math.floor(total_iterations*recall_topk*0.01)
                    # topk_idx = np.argsort(predictions_all)[-recall_topk:]
                else:
                    recall_topk = int(score_type)
                    # topk_idx = np.argsort(predictions_all)[-recall_topk:]
    
                # re-ranking
                # sampling for score_type
                # num_sample = 842 - recall_topk
                # top_remaining_idx = list((set(topz_idx) - set(topk_idx)))
                # sampled = np.array(random.sample(top_remaining_idx, num_sample))
                # recall_topk_new = np.concatenate((topk_idx, sampled))
                recall_topk_new = topz_idx
                
                # get global similarities
                global_similarities = predictions_all[recall_topk_new]
                # get local similarities
                local_similarities = []
                for idx in recall_topk_new:
                    src_emb = src_embs[idx]
                    ref_emb = ref_embs[idx]
                    
                    start_idx = np.sum(obj_count[:idx])
                    end_idx_src = start_idx + obj_count[idx][0]
                    end_idx_ref = end_idx_src + obj_count[idx][1]
    
                    src_global_ids = global_ids_all[start_idx:end_idx_src]
                    ref_global_ids = global_ids_all[end_idx_src:end_idx_ref]
                    
                    # sinkhorn, local similarity
                    local_similarity = self.customized_sinkhorn(src_emb, 
                                                                   src_global_ids, 
                                                                   ref_emb, 
                                                                   ref_global_ids)
                    local_similarities.append(local_similarity)
                    
                local_similarities = np.array(local_similarities)
                # because global similarity tends to be large, so we choose the ratio 3:7 instead 5:5
                similarities = self.similarity_fusion(global_similarities, local_similarities)
                # top_k_idx_new = np.argmax(similarities)
                top_k_idx_new = np.argsort(similarities)[-recall_topk:][::-1]
                top_k_idx = np.array(recall_topk_new[top_k_idx_new])
                if score_type == "1":
                    t2 = time.time()
                    delta_t = t2 - t1
                    delta_ts.append(delta_t)        
                elif score_type == '5' and src_scan_id == '41385849-a238-2435-81d0-ceb0eba4541a_4':
                    num_positives = np.sum(target_all)
                    top_5_ids = ref_ids[top_k_idx]
                    src_scan_id = src_scan_id.reshape(1,)
                    top_5_ids = np.concatenate((src_scan_id, top_5_ids))
                    # if min(top_5_idx) <= (num_positives - 1):
                    print(src_scan_id)
                    print(top_k_idx)
                    print(top_5_ids)
                    print('Hello')
                num_positives = np.sum(target_all) # number of positives of the query
                if min(top_k_idx) <= (num_positives - 1):
                    self.scores[score_type].append(1)
                else:
                    self.scores[score_type].append(0)
            
                    
     
        for recall_type, recall_list in self.scores.items():
            recall = sum(recall_list)/len(recall_list)
            results_dict['alignment_metrics']['pr_recall'][recall_type] = recall
        
        self.after_test_epoch()
        print("time for reranking: ", mean(delta_ts))
        self.print_metrics(results_dict)
        
        
    def similarity_fusion(self, global_similarities, local_similarities, use_condition=False):

        if use_condition:
            # Calculate the absolute difference and apply the condition
            condition = global_similarities - local_similarities >= 0.5

            # Set the values to zero where the condition is True
            global_similarities[condition] = 0
            local_similarities[condition] = 0

        # Calculate similarities
        similarities = 0.0 * global_similarities + 1.0 * local_similarities
        return similarities
        
    def sinkhorn(self, cost_matrix, lambda_reg=0.1, num_iters=100):
        K = np.exp(cost_matrix / lambda_reg)
        u = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
        for _ in range(num_iters):
            v = 1.0 / (K.T @ u)
            u = 1.0 / (K @ v)
        return u[:, None] * K * v[None, :]

        
    def resolve_label_ambiguity(self, label, labels_graph1, labels_graph2, features_g1, features_g2, metric_typ='cosine'):
        indices_g1 = np.where(labels_graph1 == label)[0]
        indices_g2 = np.where(labels_graph2 == label)[0]
        features_label_g1 = features_g1[indices_g1]
        features_label_g2 = features_g2[indices_g2]
        similarities = cosine_similarity(features_label_g1, features_label_g2)

        if metric_typ == 'sinkhorn':
            cost_matrix_label = 1 - similarities
            sinkhorn_matrix_label = self.sinkhorn(cost_matrix_label)
            assignments_label = list(self.find_max_values_and_indices(sinkhorn_matrix_label))
        elif metric_typ == 'cosine':
            assignments_label = self.find_max_values_and_indices(similarities)
        rows, cols = zip(*assignments_label)
        extracted_elements = similarities[rows, cols]
        return extracted_elements
    
    def find_max_values_and_indices(self, X):
        original_shape = X.shape
        original_X = np.copy(X)
        if original_shape[0] == 1:
            return [(0, np.argmax(X))]
        elif original_shape[1] == 1:
            return [(np.argmax(X), 0)]
        
        max_indices = []
        
        # Keep a track of removed rows and columns
        removed_rows = []
        removed_cols = []
        
        # while len(X.shape) > 1 and np.prod(X.shape) > 1:  # Ensure X is not a 1D array/vector and not empty
        while X.shape[0] != 1 and X.shape[1] != 1:
            # Find max and its index in the current matrix
            max_val_index = np.unravel_index(np.argmax(X), X.shape)
            
            # Calculate original indices
            original_row = max_val_index[0] + len([r for r in removed_rows if r <= max_val_index[0]])
            original_col = max_val_index[1] + len([c for c in removed_cols if c <= max_val_index[1]])
            max_indices.append((original_row, original_col))
            
            # Update removed rows and columns
            removed_rows.append(original_row)
            removed_cols.append(original_col)
            
            # Remove the row and column of the max value
            X = np.delete(X, max_val_index[0], axis=0)
            X = np.delete(X, max_val_index[1], axis=1)
        
        # Process the final element if X is a 1D array
        if original_shape[0] != original_shape[1]:
            if original_shape[0] - len(removed_rows) == 1:  # Last element is in the remaining row
                row_index = [r for r in range(original_shape[0]) if r not in removed_rows][0]
                max_value = np.max(X)
                indices = np.where(original_X == max_value)

                # Handling based on the dimensionality of the array
                if original_X.ndim == 1:
                    # For 1D arrays, we just need the index directly
                    col_index = indices[0][0]
                elif original_X.ndim == 2:
                    # For 2D arrays, we pick the row index of the first occurrence
                    col_index = indices[0][0]  # indices[0] contains row indices, indices[1] contains column indices

                # col_index = int(np.where(original_X == max_value)[1])
                max_indices.append((row_index, col_index))
            else:  # Last element is in the remaining column
                col_index = [c for c in range(original_shape[1]) if c not in removed_cols][0]
                max_value = np.max(X)
                indices = np.where(original_X == max_value)

                # Handling based on the dimensionality of the array
                if original_X.ndim == 1:
                    # For 1D arrays, we just need the index directly
                    row_index = indices[0][0]
                elif original_X.ndim == 2:
                    # For 2D arrays, we pick the row index of the first occurrence
                    row_index = indices[0][0]  # indices[0] contains row indices, indices[1] contains column indices

                # row_index = int(np.where(original_X == max_value)[0])
                max_indices.append((row_index, col_index))
        else:
            row_index = [r for r in range(original_shape[0]) if r not in removed_rows][0]
            col_index = list(set(range(original_shape[1])) - set(removed_cols))
            max_indices.append((row_index, col_index[0]))
            
        return max_indices        
                
        
    def get_overlap(self, src_global_ids, ref_global_ids):
        unique1, counts1 = np.unique(src_global_ids, return_counts=True)
        unique2, counts2 = np.unique(ref_global_ids, return_counts=True)

        # Convert to dictionary for easier manipulation
        count_dict1 = dict(zip(unique1, counts1))
        count_dict2 = dict(zip(unique2, counts2))

        # Determine the intersection and union counts
        intersection_counts_np = {element: min(count_dict1.get(element, 0), count_dict2.get(element, 0)) for element in set(count_dict1) | set(count_dict2)}
        union_counts_np = {element: max(count_dict1.get(element, 0), count_dict2.get(element, 0)) for element in set(count_dict1) | set(count_dict2)}

        # Sum of intersection and union counts
        intersection_sum_np = sum(intersection_counts_np.values())
        union_sum_np = sum(union_counts_np.values())

        # Calculate the intersect ratio
        return intersection_sum_np / union_sum_np

    
    def customized_sinkhorn(self, src_emb, src_global_ids, ref_emb, ref_global_ids):
        src_emb = src_emb.squeeze()
        ref_emb = ref_emb.squeeze()
        unique_labels = np.intersect1d(src_global_ids, ref_global_ids)  # Labels present in both graphs
        overlap = self.get_overlap(src_global_ids, ref_global_ids)
        cosine_similarity_scores = []
        for label in unique_labels:
            local_similarity = self.resolve_label_ambiguity(label, src_global_ids, ref_global_ids, src_emb, ref_emb)
            cosine_similarity_scores.extend(local_similarity)
        if len(cosine_similarity_scores) != 0:
            local_similarity = (sum(cosine_similarity_scores)/len(cosine_similarity_scores))*overlap
        else:
            local_similarity = 0
        return local_similarity
        


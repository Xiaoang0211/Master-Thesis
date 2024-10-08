import numpy as np
import json
import math
from texttable import Texttable
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import itertools

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
    print(t.draw())
    
def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def calculate_normalized_ged(data):
    """
    Calculating the normalized GED for a pair of graphs.
    :param data: Data table.
    :return norm_ged: Normalized GED score.
    """
    norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
    return norm_ged

def load_paires(file,graph_pairs_dir):
  paires=[]
  with open(file) as f:
    while True:
      line=f.readline()
      if not line:
        break
      line=line.strip().split(" ")
      paires.append([os.path.join(graph_pairs_dir,line[0]),os.path.join(graph_pairs_dir,line[1])])
  return paires

def load_paires_json(file):
  with open(file) as f:
    anchor_pair_list = json.load(f)
  return anchor_pair_list


def listDir(path, list_name):
    """
    :param path: root_dir
    :param list_name: abs paths of all files under the root_dir
    :return:
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listDir(file_path, list_name)
        else:
            list_name.append(file_path)

def flip_point_cloud(batch_data):
    if random.random() >0.5:
        batch_data[:,:,0] = -batch_data[:,:,0]
    return batch_data

def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    # along y
    # rotation_matrix = np.array([[cosval, 0, sinval],
    #               [0, 1, 0],
    #               [-sinval, 0, cosval]])
    # shape_pc = batch_data[k, ...]
    # along z
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data

def gen_random_trans(x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), z_range=(-0.1, 0.1)):
  """ Generate random transformation for 3D points

  Args:
      trans_range (_type_): _description_
      rot_range (_type_): _description_
  """
  t_x = np.random.uniform(x_range[0], x_range[1])
  t_y = np.random.uniform(y_range[0], y_range[1])
  t_z = np.random.uniform(z_range[0], z_range[1])
    
  # Construct the translation matrix
  translation_matrix = np.array([
        [t_x],
        [t_y],
        [t_z],
    ])
  return translation_matrix

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
  """
  B, N, C = batch_data.shape
  assert(clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_data += batch_data
  return jittered_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.015, angle_clip=0.045): # angle_sigma=0.06, angle_clip=0.18
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in range(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data

def shift_point_cloud(batch_data, shift_range=0.3): # 0.1
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,3))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data
    
def load_txt_list(training_graphs_dir): 
  with open(training_graphs_dir, 'r') as file:
    all_subscans = [all_subscans.strip() for all_subscans in file]
  return all_subscans
    

  
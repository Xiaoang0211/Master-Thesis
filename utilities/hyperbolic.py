import torch
import time

# Ensure you have a CUDA-capable GPU and PyTorch with GPU support installed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 800
feat_dim = 256

# Random sample data tensors (batches of points)
source = torch.rand((batch_size, feat_dim, 1)).to(device) * 0.999  # Multiplied by 0.999 to ensure points are inside the Poincaré ball
target = torch.rand((batch_size, feat_dim, 1)).to(device) * 0.999

# Hyperbolic distance function for batches
def hyperbolic_distance(u, v):
    norm_diff = torch.norm(u - v, dim=1, p=2)**2  # Compute squared L2 norm across feature dimension
    denom_u = 1 - torch.norm(u, dim=1, p=2)**2
    denom_v = 1 - torch.norm(v, dim=1, p=2)**2
    
    distance = torch.acosh(1 + 2 * norm_diff / (denom_u * denom_v)).squeeze()  # .squeeze() is to remove the last dimension, making the shape just (batch_size,)
    return distance
t0 = time.time()
distances = hyperbolic_distance(source, target)
t1 = time.time()
print(t1-t0)
print(distances.shape)  # Expected: torch.Size([400])

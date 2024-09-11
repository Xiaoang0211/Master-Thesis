import torch
import torch.nn as nn
import torch.nn.functional as F



def preprocess_predictions(num_pos_per_query, num_neg_per_query, prediction, batch_size, loss_typ):
    # reshape the preidction to (batch_size, num_pairs_per_query)
    if loss_typ == "quadruplet":
        num_pairs_per_query = num_pos_per_query + num_neg_per_query*2 # (2 pos 10 neg and 10 other neg)
        prediction_reshaped = prediction.reshape(batch_size, num_pairs_per_query)
        prediction_pos = prediction_reshaped[:, :num_pos_per_query]
        prediction_neg = prediction_reshaped[:, num_pos_per_query:num_pos_per_query + num_neg_per_query]
        prediction_other_neg = prediction_reshaped[:, num_pos_per_query + num_neg_per_query:]
        return prediction_pos, prediction_neg, prediction_other_neg
    else:
        num_pairs_per_query = num_pos_per_query + num_neg_per_query # (2 pos 10 neg)
        prediction_reshaped = prediction.reshape(batch_size, num_pairs_per_query)
        prediction_pos = prediction_reshaped[:, :num_pos_per_query]
        prediction_neg = prediction_reshaped[:, num_pos_per_query:num_pos_per_query + num_neg_per_query]
        return prediction_pos, prediction_neg
    

class GraphMatchLoss(nn.Module):
    def __init__(self, loss_type, num_pos_per_query, num_neg_per_query, batch_size, margin1=0.5, margin2=0.5, temperature=0.1):
        super(GraphMatchLoss, self).__init__()
        self.loss_type = loss_type

        # Create an instance of the appropriate loss module
        if loss_type == "triplet":
            self.loss_module = TripletLoss(num_pos_per_query, num_neg_per_query, batch_size, margin1)
        elif loss_type == "quadruplet":
            self.loss_module = QuadrupletLoss(num_pos_per_query, num_neg_per_query, batch_size, margin1, margin2)
        elif loss_type == "HPHN_quadruplet":
            self.loss_module = HPHNQuadrupletLoss(num_pos_per_query, num_neg_per_query, batch_size, margin1)
        elif loss_type == "contrastive":
            self.loss_module = ContrastiveLoss(num_pos_per_query, num_neg_per_query, batch_size, margin1)
        elif loss_type == "binary_cross_entropy":
            self.loss_module = None  # Binary cross entropy does not require a separate module
        elif loss_type == "nt_xent":
            self.loss_module = NTXentLoss(num_pos_per_query, num_neg_per_query, batch_size, temperature)
        else:
            raise ValueError("Undefined loss type")

    def forward(self, prediction, target):
        if self.loss_type == "binary_cross_entropy":
            if target is None:
                raise ValueError("Target must be provided for binary cross entropy loss")
            target = target.float()
            return torch.mean(F.binary_cross_entropy(prediction, target))
        else:
            return self.loss_module(prediction)

    
class TripletLoss(nn.Module):
    def __init__(self, num_pos_per_query, num_neg_per_query, batch_size, margin=0.5):
        super(TripletLoss, self).__init__()
        self.num_pos_per_query = num_pos_per_query
        self.num_neg_per_query = num_neg_per_query
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, prediction):
        prediction_pos, prediction_neg = preprocess_predictions(
                                        self.num_pos_per_query, 
                                        self.num_neg_per_query, 
                                        prediction, 
                                        self.batch_size)
    
        # Expand anchor predictions to match the number of positive and negative pairs
        prediction_pos_expanded = prediction_pos.unsqueeze(2).expand(-1, -1, self.num_neg_per_query)
        prediction_neg_expanded = prediction_neg.unsqueeze(1).expand(-1, self.num_pos_per_query, -1)
        
        # Compute differences for each triplet combination
        differences = prediction_pos_expanded - prediction_neg_expanded
        
        # Compute the triplet loss for each triplet combination
        losses = torch.relu(self.margin + differences)
        
        # Average over all the triplet loss values
        return torch.mean(losses)

class QuadrupletLoss(nn.Module):
    def __init__(self, num_pos_per_query, num_neg_per_query, batch_size, margin1=0.5, margin2=0.5):
        super(QuadrupletLoss, self).__init__()
        self.num_pos_per_query = num_pos_per_query
        self.num_neg_per_query = num_neg_per_query
        self.batch_size = batch_size
        self.margin2 = margin2
        self.triplet_loss = TripletLoss(num_pos_per_query, num_neg_per_query, batch_size, margin1)
    
    def forward(self, prediction):
        # Get the predictions for positive, negative, and other negative samples
        prediction_pos, prediction_neg, prediction_other_neg = preprocess_predictions(
                                            self.num_pos_per_query, 
                                            self.num_neg_per_query, 
                                            prediction, 
                                            self.batch_size,
                                            loss_typ="quadruplet")
        
        # Compute the triplet loss for (A, P, N)
        
        triplet_loss_value = self.triplet_loss(prediction[:, :num_pos_per_query + num_neg_per_query])
        
        # Compute differences for the other negative sample and the negatives of the actual anchor
        differences_other = prediction_other_neg - prediction_neg
        
        # Compute the loss for the (O, N) part
        losses_other = torch.relu(self.margin2 + differences_other)
        
        # Average over all the computed losses for the (O, N) part
        avg_loss_other = torch.mean(losses_other)
        
        # Combine the two losses
        total_loss = triplet_loss_value + avg_loss_other
        
        return total_loss


class HPHNQuadrupletLoss(nn.Module):
    def __init__(self, num_pos_per_query, num_neg_per_query, batch_size, margin=0.5):
        super(HPHNQuadrupletLoss, self).__init__()
        self.num_pos_per_query = num_pos_per_query
        self.num_neg_per_query = num_neg_per_query
        self.batch_size = batch_size
        self.margin = margin
        
        
    def forward(self, prediction):
        """HPHN loss for one query subscan

        Args:
            prediction (tensor): predicted similarity scores of all input pairs
            data (dict): 
        """
        # reshape the preidction to (batch_size, num_pairs_per_query)
        prediction_pos, prediction_neg, prediction_other_neg = preprocess_predictions(
                                                                self.num_pos_per_query, 
                                                                self.num_neg_per_query, 
                                                                prediction, 
                                                                self.batch_size,
                                                                loss_typ="quadruplet")
        # hard positive with min similarity
        min_similarity_pos = torch.min(prediction_pos, dim=1, keepdim=True).values
        
        # hard negative with max similarity
        max_similarity_neg = torch.max(prediction_neg, dim=1, keepdim=True).values
        
        # the other negative
        max_other_neg = torch.max(prediction_other_neg, dim=1, keepdim=True).values
        similarity_hn = torch.max(torch.cat((max_similarity_neg, max_other_neg), dim=1), dim=1).values
        similarity_hn_reshaped = similarity_hn.reshape(batch_size, 1)
        
        # hinge loss
        thres = similarity_hn_reshaped - min_similarity_pos + self.margin
        zeros = torch.zeros(batch_size, 1).cuda()

        L_HPHN = torch.max(torch.cat((thres, zeros), dim=1), dim=1).values
        # L_HPHN = torch.mean(L_HPHN)
        return L_HPHN

class ContrastiveLoss(nn.Module):
    def __init__(self, num_pos_per_query, num_neg_per_query, batch_size, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.num_pos_per_query = num_pos_per_query
        self.num_neg_per_query = num_neg_per_query
        self.batch_size = batch_size
        self.margin = margin
        
    def forward(self, prediction):
        prediction_pos, prediction_neg = preprocess_predictions(
                                    self.num_pos_per_query, 
                                    self.num_neg_per_query, 
                                    prediction, 
                                    self.batch_size,
                                    loss_typ="contrastive")
    
        # Compute the loss for positive (similar) pairs
        positive_loss = torch.mean((1 - prediction_pos)**2)
        
        # Compute the loss for negative (dissimilar) pairs
        zeros_tensor = torch.zeros_like(prediction_neg)
        negative_loss = torch.mean(torch.max(zeros_tensor, self.margin - prediction_neg)**2)
        
        # Combine the two losses
        total_loss = 0.5 * (positive_loss + negative_loss)
        
        return total_loss

class NTXentLoss(nn.Module):
    def __init__(self, num_pos_per_query, num_neg_per_query, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.num_pos_per_query = num_pos_per_query
        self.num_neg_per_query = num_neg_per_query
        self.batch_size = batch_size
        self.temperature = temperature
        
    def forward(self, prediction):
        """
        Compute the NT-Xent loss from provided similarity scores in a vectorized manner.
        
        Args:
        - prediction (torch.Tensor): 1D tensor containing similarity scores
        - num_pos_per_query (int): Number of positive samples per query
        - num_neg_per_query (int): Number of negative samples per query
        - batch_size (int): Number of samples in the batch
        - temperature (float): Temperature parameter for scaling similarities
        
        Returns:
        - loss (torch.Tensor): NT-Xent loss value
        """
        # Reshape the similarity scores
        prediction_pos, prediction_neg = preprocess_predictions(self.num_pos_per_query, 
                                                                self.num_neg_per_query, 
                                                                prediction, 
                                                                self.batch_size, 
                                                                "NTXent")
        prediction_pos_expanded = prediction_pos.unsqueeze(2).expand(-1, -1, 1)
        prediction_neg_expanded = prediction_neg.unsqueeze(1).expand(-1, self.num_pos_per_query, -1)
        # Compute the numerators of the NT-Xent loss for all samples in the batch
        numerators = torch.exp(prediction_pos_expanded / self.temperature)
        total_scores = torch.cat([prediction_pos_expanded, prediction_neg_expanded], dim=2)
        denominators = torch.sum(torch.exp(total_scores / self.temperature), dim=2, keepdim=True)
        loss_values = -torch.log(numerators / denominators).squeeze(-1)
        loss = torch.mean(loss_values)
        return loss

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 5
    num_pos_per_query = 2
    num_neg_per_query = 10
    prediction_dim = (batch_size, num_pos_per_query + num_neg_per_query)
    
    # Move the tensor to the device (GPU if available, otherwise CPU)
    prediction = torch.rand(prediction_dim) #.to(device)
    
    import time
    t0 = time.time()
    # Calculate the loss on the device
    # loss = nt_xent_loss(prediction, num_pos_per_query, num_neg_per_query, batch_size)
    # t1 = time.time()
    # print("t1 - t0: ", t1 - t0)
    # print(loss)
import torch

def preprocess_predictions_for_trip_quad(num_pos_per_query, num_neg_per_query, prediction, batch_size, loss_typ="triplet"):
    # reshape the preidction to (batch_size, num_pairs_per_query)
    if loss_typ == "triplet":
        num_pairs_per_query = num_pos_per_query + num_neg_per_query # (2 pos 10 neg)
        prediction_reshaped = prediction.reshape(batch_size, num_pairs_per_query)
        prediction_pos = prediction_reshaped[:, :num_pos_per_query]
        prediction_neg = prediction_reshaped[:, num_pos_per_query:num_pos_per_query + num_neg_per_query]
        return prediction_pos, prediction_neg
    elif loss_typ == "quadruplet":
        num_pairs_per_query = num_pos_per_query + num_neg_per_query*2 # (2 pos 10 neg and 10 other neg)
        prediction_reshaped = prediction.reshape(batch_size, num_pairs_per_query)
        prediction_pos = prediction_reshaped[:, :num_pos_per_query]
        prediction_neg = prediction_reshaped[:, num_pos_per_query:num_pos_per_query + num_neg_per_query]
        prediction_other_neg = prediction_reshaped[:, num_pos_per_query + num_neg_per_query:]
        return prediction_pos, prediction_neg, prediction_other_neg
    
def triplet_loss(num_pos_per_query, num_neg_per_query, prediction, batch_size, margin=0.5):
    # Get the predictions for positive and negative samples
    prediction_pos, prediction_neg = preprocess_predictions_for_trip_quad(
                                        num_pos_per_query, 
                                        num_neg_per_query, 
                                        prediction, 
                                        batch_size)
    
    # Expand anchor predictions to match the number of positive and negative pairs
    prediction_pos_expanded = prediction_pos.unsqueeze(2).expand(-1, -1, num_neg_per_query)
    prediction_neg_expanded = prediction_neg.unsqueeze(1).expand(-1, num_pos_per_query, -1)
    
    # Compute differences for each triplet combination
    differences = prediction_pos_expanded - prediction_neg_expanded
    
    # Compute the triplet loss for each triplet combination
    losses = torch.relu(margin + differences)
    
    # Average over all the triplet loss values
    return torch.mean(losses)


def quadruplet_loss(num_pos_per_query, num_neg_per_query, prediction, batch_size, margin1=0.5, margin2=0.5):
    # Get the predictions for positive, negative, and other negative samples
    prediction_pos, prediction_neg, prediction_other_neg = preprocess_predictions_for_trip_quad(
                                        num_pos_per_query, 
                                        num_neg_per_query, 
                                        prediction, 
                                        batch_size,
                                        loss_typ="quadruplet")
    
    # Compute the triplet loss for (A, P, N)
    triplet_loss_value = triplet_loss(num_pos_per_query, num_neg_per_query, prediction[:, :num_pos_per_query + num_neg_per_query], batch_size, margin=margin1)
    
    # Compute differences for the other negative sample and the negatives of the actual anchor
    differences_other = prediction_other_neg - prediction_neg
    
    # Compute the loss for the (O, N) part
    losses_other = torch.relu(margin2 + differences_other)
    
    # Average over all the computed losses for the (O, N) part
    avg_loss_other = torch.mean(losses_other)
    
    # Combine the two losses
    total_loss = triplet_loss_value + avg_loss_other
    
    return total_loss


def HPHN_quadruplet_loss(num_pos_per_query, num_neg_per_query, prediction, batch_size, margin=0.5):
    """HPHN loss for one query subscan

    Args:
        prediction (tensor): predicted similarity scores of all input pairs
        data (dict): 
    """
    # reshape the preidction to (batch_size, num_pairs_per_query)
    prediction_pos, prediction_neg, prediction_other_neg = preprocess_predictions_for_trip_quad(
                                                            num_pos_per_query, 
                                                            num_neg_per_query, 
                                                            prediction, 
                                                            batch_size,
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
    thres = similarity_hn_reshaped - min_similarity_pos + margin
    zeros = torch.zeros(batch_size, 1).cuda()

    L_HPHN = torch.max(torch.cat((thres, zeros), dim=1), dim=1).values
    # L_HPHN = torch.mean(L_HPHN)
    return L_HPHN


def contrative_loss(num_pos_per_query, num_neg_per_query, prediction, batch_size, margin=0.5):
    prediction_pos, prediction_neg = preprocess_predictions_for_trip_quad(
                                    num_pos_per_query, 
                                    num_neg_per_query, 
                                    prediction, 
                                    batch_size)
    
    # Compute the loss for positive (similar) pairs
    positive_loss = torch.mean((1 - prediction_pos)**2)
    
    # Compute the loss for negative (dissimilar) pairs
    zeros_tensor = torch.zeros_like(prediction_neg)
    negative_loss = torch.mean(torch.max(zeros_tensor, margin - prediction_neg)**2)
    
    # Combine the two losses
    total_loss = 0.5 * (positive_loss + negative_loss)
    
    return total_loss
    

def nt_xent_loss(prediction, num_pos_per_query, num_neg_per_query, batch_size, temperature=0.5):
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
    prediction_pos, prediction_neg = preprocess_predictions_for_trip_quad(num_pos_per_query, num_neg_per_query, prediction, batch_size, "triplet")
    prediction_pos_expanded = prediction_pos.unsqueeze(2).expand(-1, -1, 1)
    prediction_neg_expanded = prediction_neg.unsqueeze(1).expand(-1, num_pos_per_query, -1)
    # Compute the numerators of the NT-Xent loss for all samples in the batch
    numerators = torch.exp(prediction_pos_expanded / temperature)
    total_scores = torch.cat([prediction_pos_expanded, prediction_neg_expanded], dim=2)
    denominators = torch.sum(torch.exp(total_scores / temperature), dim=2, keepdim=True)
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
    loss = nt_xent_loss(prediction, num_pos_per_query, num_neg_per_query, batch_size)
    t1 = time.time()
    print("t1 - t0: ", t1 - t0)
    print(loss)
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.networks.neural_tensor_networks import GeM

def generate_mask(src, tgt):
    src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    # src_mask = src != 0
    # tgt_mask = tgt != 0
    # src_pad_node = src[0,:, 29]
    # seq_length = tgt.size(1)
    # device = src.device
    # nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()  # Create the tensor on the same device as src    nopeak_mask = nopeak_mask.unsqueeze(0)
    # tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask


class NodeMultiHeadCrossAttention(nn.Module):
    def __init__(self, feature_dim_out, num_heads):
        super(NodeMultiHeadCrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim_out
        self.head_dim = feature_dim_out
        
        # Linear layers for Query, Key, Value for all heads
        self.query = nn.Linear(feature_dim_out, feature_dim_out * num_heads)
        self.key = nn.Linear(feature_dim_out, feature_dim_out * num_heads)
        self.value = nn.Linear(feature_dim_out, feature_dim_out * num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim_out * num_heads, feature_dim_out)

    def forward(self, source, target, src_mask=None, tgt_mask=None):
        """
        source: Tensor of shape (batch_size, source_node_num, feature_dim)
        target: Tensor of shape (batch_size, target_node_num, feature_dim)
        """
        # # when cross att is before dgcnn
        # batch_size, _, source_node_num = source.size()
        # _, _, target_node_num = target.size()
        
        # when cross att is after dgcnn
        batch_size, source_node_num, _ = source.size()
        _, target_node_num, _ = target.size()
        
        # Transform source and target using Q, K, V linear layers
        Q = self.query(source).view(batch_size, source_node_num, self.num_heads, self.feature_dim).transpose(1, 2)
        K = self.key(target).view(batch_size, target_node_num, self.num_heads, self.feature_dim).transpose(1, 2)
        V = self.value(target).view(batch_size, target_node_num, self.num_heads, self.feature_dim).transpose(1, 2)
        
        # Compute attention 
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))/(self.head_dim**0.5)
        
        # Apply masks
        if src_mask is not None:
            attention_scores = attention_scores.masked_fill(src_mask == 0, float('-inf'))
        
        if tgt_mask is not None:
            attention_scores = attention_scores.masked_fill(tgt_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads across the node dimension
        weighted_values_concat = weighted_values.transpose(2, 1).contiguous().view(batch_size, source_node_num, -1)

        # Project concatenated output to original feature dimension
        output = self.out_proj(weighted_values_concat)
        
        return output


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, ff_dim, dropout_rate):
        super(CrossAttention, self).__init__()
        
        self.node_cross_attention = NodeMultiHeadCrossAttention(feature_dim, num_heads)
        
        # Layer normalization
        self.norm0 = nn.LayerNorm(feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Feed-forward neural network
        self.ff_source = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_dim)
        )
        
        self.ff_target = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_dim)
        )
        
        
    def forward(self, source, target, src_mask = None, tgt_mask = None):
        """
        source: Tensor of shape (batch_size, source_node_num, feature_dim)
        target: Tensor of shape (batch_size*12, target_node_num, feature_dim)
        """
        
        # node cross attention
        src_mask, tgt_mask = generate_mask(source, target)
         
        attention_output_source = self.node_cross_attention(source, target, src_mask, tgt_mask)
        attention_output_target = self.node_cross_attention(target, source, tgt_mask, src_mask)
        
        # residual & norm
        source = self.norm1(source + self.dropout(attention_output_source))
        target = self.norm1(target + self.dropout(attention_output_target))
        
        # Feed-forward network with residual connection and dropout
        ff_output_source = self.ff_source(source)
        ff_output_target = self.ff_target(target)
        
        # residual & norm
        output_source = self.norm2(source + self.dropout(ff_output_source))
        output_target = self.norm2(target + self.dropout(ff_output_target))
        
        return output_source, output_target

class StackedCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers=2, ff_dim=512, dropout_rate=0.1):
        super(StackedCrossAttention, self).__init__()
        
        self.layers = nn.ModuleList([CrossAttention(feature_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, source, target):
        """
        source: Tensor of shape (batch_size, source_node_num, feature_dim)
        target: Tensor of shape (batch_size, target_node_num, feature_dim)
        """
        for layer in self.layers:
            source, target = layer(source, target)
        
        return source, target

class NodeSelfAttention(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, feature_dim_out, num_heads):
        """
        :param args: Arguments object.
        """
        super(NodeSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim_out
        self.head_dim = feature_dim_out
        
        self.query = nn.Linear(feature_dim_out, feature_dim_out * num_heads)
        self.key = nn.Linear(feature_dim_out, feature_dim_out * num_heads)
        self.value = nn.Linear(feature_dim_out, feature_dim_out * num_heads)
        
        self.out_proj = nn.Linear(feature_dim_out*num_heads, feature_dim_out)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation using QKV attention.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector. 
        """

        batch_size, num_nodes, _ = embedding.size()
        Q = self.query(embedding).view(batch_size, num_nodes, self.num_heads, self.feature_dim).transpose(1,2)
        K = self.query(embedding).view(batch_size, num_nodes, self.num_heads, self.feature_dim).transpose(1,2)
        V = self.query(embedding).view(batch_size, num_nodes, self.num_heads, self.feature_dim).transpose(1,2)
     
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))/(self.head_dim**0.5) # bxnxf bxfxn -> bxnxn
        attention_weights = F.softmax(attention_scores, dim=-1) # scale
        weighted_values = torch.matmul(attention_weights, V) # bxnxn bxnxf -> bxnxf
        
        weighted_values_concat = weighted_values.transpose(2, 1).contiguous().view(batch_size, num_nodes, -1)
        
        output = self.out_proj(weighted_values_concat)
        
        return output
    
    
class SelfAttentionWithFNNAndDropout(nn.Module):
    def __init__(self, feature_dim, num_heads, ff_dim, dropout_rate):
        super(SelfAttentionWithFNNAndDropout, self).__init__()
        
        self.node_self_attention = NodeSelfAttention(feature_dim, num_heads)
        
        # Layer normalization
        self.norm0 = nn.LayerNorm(feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Feed-forward neural network
        self.ff_source = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_dim)
        )
        
        self.ff_target = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_dim)
        )
        
        
    def forward(self, embedding):
        """
        source: Tensor of shape (batch_size, source_node_num, feature_dim)
        target: Tensor of shape (batch_size*12, target_node_num, feature_dim)
        """
        
        # Cross attention with residual connection and dropout
    
        # node self attention
        attention_output_source = self.node_self_attention(embedding)
        
        # residual & norm
        embedding = self.norm1(embedding + self.dropout(attention_output_source))
        
        # Feed-forward network with residual connection and dropout
        ff_output_source = self.ff_source(embedding)

        
        # residual & norm
        output_embedding = self.norm2(embedding + self.dropout(ff_output_source))
        
        return output_embedding
    
class StackedSelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers=2, ff_dim=512, dropout_rate=0.1):
        super(StackedSelfAttention, self).__init__()
        self.linear = nn.Linear(feature_dim, 1)  # Final layer to project to the desired output dimension
        self.layers = nn.ModuleList([SelfAttentionWithFNNAndDropout(feature_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)])
        self.GeM_pooling = GeM()
        
    def forward(self, embedding):
        """
        source: Tensor of shape (batch_size, source_node_num, feature_dim)
        target: Tensor of shape (batch_size, target_node_num, feature_dim)
        """
        for layer in self.layers:
            embedding = layer(embedding)
        embedding = self.GeM_pooling(embedding).squeeze(-1)
        output = self.linear(embedding).squeeze()  # Linear layer and squeeze to get the final output shape
        return output
    
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Normalize the input
        src2 = self.norm1(src)
        # Apply self-attention to the normalized input
        attn_output = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)[0]
        # Add the output of self-attention to the normalized input (src2) and apply dropout
        src = src2 + self.dropout1(attn_output)  # Correction here: add to src2
        
        # Feed-forward network operations remain the same
        src2 = self.norm2(src)  # Normalizing the input for the feed-forward network
        src2 = F.relu(self.linear1(src2))
        src2 = self.dropout2(src2)
        src2 = self.linear2(src2)
        src = src + src2  # Adding the feed-forward output to the input of the feed-forward network
        
        return src

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, 1)  # Final layer to project to the desired output dimension
        self.GeM_pooling = GeM()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)
        # src = src.mean(dim=1)  # Mean pooling over the sequence length
        # src = self.GeM_pooling(src).squeeze(-1)
        # output = self.linear(src)  # Linear layer and squeeze to get the final output shape
        # output = torch.sigmoid(output)
        return src[:,:30,:], src[:,30:,:]
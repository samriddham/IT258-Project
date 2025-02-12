import torch
import torch.nn as nn

'''
Attentive GRU
Takes in raw week information and then makes Ai for all of the weeks for all the stocks (if bacthed)
'''

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: torch.tensor (batch_size, time_step, hidden_dim)
        """
        attention_scores = self.fc(x)  # Shape = (batch_size, time_step, 1)
        attention_scores = self.tanh(attention_scores)
        attention_weights = self.softmax(attention_scores)  # Shape = (batch_size, time_step, 1)
        return attention_weights

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention_block = AttentionBlock(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.hidden_dim = hidden_dim
    
    def forward(self, seq):
        """
        seq: torch.tensor (batch_size, time_step, input_dim)
        """
        seq_vector, _ = self.gru1(seq)  # Shape = (batch_size, time_step, hidden_dim)
        seq_vector, _ = self.gru2(seq_vector) # Shape = (batch_size, time_step, hidden_dim)
        seq_vector = self.dropout(seq_vector)
        
        attention_weight = self.attention_block(seq_vector)  # Shape = (batch_size, timestep, 1)
        Ai = seq_vector * attention_weight  # Shape = (batch_size, time_step, hidden_dim)
        Ai = torch.sum(Ai, dim=1)  # Shape = (batch_size, hidden_dim)
        
        return Ai
    
# TODO Make the Adjacency Matrix of everything. Update = This is Done

# OK heres how im going to impelemnt it if u have a better version please come ahead why am i even writing this is its not like anyones going to read this anyway 
# welp just in case you do read it this is how im making it im making a string dict whre the key is the node and the values are the neighbours
# then when we want to acess the Ai list for each stcok we can just use a HashMap to do so.
import os
import pandas as pd
sector_mat = dict() ### GLOBAL VAR

file_path = r'./Preprocessed_data'
for file in os.listdir(file_path):
    df = pd.read_csv(file_path+"/"+file)
    # print(df.info)
    sector = df['Sector_Encoded'][0]
    if sector in sector_mat:
        sector_mat[sector].append(file[:-9])
    else:
        sector_mat[sector] = [file[:-9]]

print(sector_mat)

# TODO Create the Graphs that is going to create the sector graphs and and then we need to create these graphs for like
# Ai and Gi also like per week
# Update = Made it. Excpet the HasMap to link this to the Ai list and the Gi list cause we dont have that yet.

name_matrix = dict()
for same_sector_stocks in sector_mat.values():
    for stock_key in same_sector_stocks:
        name_matrix[stock_key] = []
        for stock_value in same_sector_stocks:
            if stock_value != stock_key:
                name_matrix[stock_key].append(stock_value)

for key,value in name_matrix.items():
    print(key,value)

# TODO Create HashMap to acess the Ai list for the given names
import torch
import torch.nn as nn

class GAT_Attention_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT_Attention_Block, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.u = nn.Parameter(torch.randn(out_dim * 2))
    
    def forward(self, ai_sq, ai_sn_list):
        ai_sq_proj = self.W(ai_sq)
        Beta = []
        for ai_sn in ai_sn_list:
            ai_sn_proj = self.W(ai_sn)
            concated = torch.concat((ai_sq_proj, ai_sn_proj), dim=1)
            mult = torch.matmul(self.u, concated.T)  # Resulting shape: (1, 1234)
            Beta.append(torch.exp(self.leaky_relu(mult)))
        
        Beta = torch.stack(Beta)  # Stack the list of tensors
        sum_beta = torch.sum(Beta, axis=0)
        output = torch.div(Beta, sum_beta)
        return output

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GAT, self).__init__()
        self.attention_block = GAT_Attention_Block(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.W1 = nn.Linear(in_dim, out_dim) # Im not sure of this in and out dim if you run into an error here please fix it or show it to saksham X):P
    
    def forward(self, ai_sq, ai_sn):
        ai_sn_proj = torch.stack([self.W1(ai_si) for ai_si in ai_sn])  # Shape = (len(ai_sn), out_dim)
        B = self.attention_block(ai_sq, ai_sn)  # Shape = (len(ai_sn),)
        B = B.unsqueeze(-1)  # Reshape B to (len(ai_sn), 1) for broadcasting
        ai_sn_weighted = B * ai_sn_proj  # Element-wise multiplication, Shape = (len(ai_sn), out_dim)
        ai_sn_sum = torch.sum(ai_sn_weighted, dim=0)  # Shape = (out_dim,)
        g_sq = self.relu(ai_sn_sum)  # Shape = (out_dim,)
        return g_sq
    
import torch.nn as nn
class AttentionBlockLTSL(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_step):
        super(AttentionBlockLTSL, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim))  # Weight matrix for u
        self.V = nn.Parameter(torch.randn(hidden_dim, 1))  # Weight matrix for transformed u
        
        self.time_step = time_step  # Length of the sequence (t)
        
    def forward(self, U):
        u_transformed = torch.tanh(torch.matmul(U, self.W))  # Shape = (batch_size, time_step, hidden_dim)
        attention_scores = torch.matmul(u_transformed, self.V)  # Shape = (batch_size, time_step, 1)
        attention_scores = torch.squeeze(attention_scores, dim=-1)  # Shape = (batch_size, time_step)
        exp_scores = torch.exp(attention_scores)  # Shape = (batch_size, time_step)
        attention_weights = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)  # Shape = (batch_size, time_step)
        weighted_U = U * attention_weights.unsqueeze(-1)  # Shape = (batch_size, time_step, input_dim)
        output = torch.sum(weighted_U, dim=1)  # Shape = (batch_size, input_dim)
        
        return output, attention_weights

# Long Term Sequential Learning with Short Term Embeddings
# Input is the Ai of all stocks and Gi of all stocks
def get_T_i(ai_sq_list, gi_sq_list, input_dim=64, hidden_size=64, t=5):
    # ai_sq_list is the list of Ai for all weeks of a stock
    # gi_sq_list is the list of Gi for all weeks of a stock
    
    T_Ai_encoder = AttentionBlock(input_dim=input_dim, hidden_dim=hidden_size, time_step=t)
    T_Gi_encoder = AttentionBlock(input_dim=input_dim, hidden_dim=hidden_size, time_step=t)
    
    T_ai_long_term = []
    T_gi_long_term = []
    
    for i in range(t, len(ai_sq_list)):
        U_Ai = torch.stack(ai_sq_list[i-t:i])  # Shape = (t, input_dim)
        U_Gi = torch.stack(gi_sq_list[i-t:i])  # Shape = (t, input_dim)
        
        aj_ai = T_Ai_encoder(U_Ai)  # Shape = (t,)
        aj_gi = T_Gi_encoder(U_Gi)  # Shape = (t,)
        
        weighted_U_Ai = aj_ai.unsqueeze(-1) * U_Ai  # Shape = (t, input_dim)
        weighted_U_Gi = aj_gi.unsqueeze(-1) * U_Gi  # Shape = (t, input_dim)
        
        T_ai_long_term.append(weighted_U_Ai.sum(dim=0))  # Shape = (input_dim,)
        T_gi_long_term.append(weighted_U_Gi.sum(dim=0))  # Shape = (input_dim,)
    
    T_ai_long_term = torch.stack(T_ai_long_term)  # Shape = (num_windows, input_dim)
    T_gi_long_term = torch.stack(T_gi_long_term)  # Shape = (num_windows, input_dim)
    
    T_A = T_ai_long_term.sum(dim=0)  # Shape = (input_dim,)
    T_G = T_gi_long_term.sum(dim=0)  # Shape = (input_dim,)
    
    return T_A, T_G

# Sector Graph

def max_pool_sector_embeddings(data):
    data = torch.stack(data)
    return torch.max(data, dim=0)[0]

def update_sector_graph(t_g_dict):
    sector_graph = {}
    for key, value in t_g_dict.items():
        sector_graph[key] = max_pool_sector_embeddings(value)
    return sector_graph

# TODO the universal map is Task 3 or 4 so im not boterhing with that right now.
# Update i guess i have to now

import torch
import torch.nn as nn
import numpy as np

class SectorEmbeddingGenerator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super(SectorEmbeddingGenerator, self).__init__()
        self.gat = GAT(in_dim=input_dim, out_dim=hidden_dim)
    
    def forward(self, max_pool_data, sector_graph):
        x = self.max_pool_sector_embeddings(max_pool_data)
        return self.gat(x, sector_graph)

class FinalLayer(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, short_term_embeddings, intra_sector_embeddings, inter_sector_embeddings):
        return self.relu(self.fc(torch.concat((short_term_embeddings, intra_sector_embeddings, inter_sector_embeddings),dim=-1)))

import torch
import torch.nn as nn
import torch.nn.functional as F

class StockPredictionLoss(nn.Module):
    def __init__(self, delta: float = 0.5, lambda_reg: float = 1e-4):
        """
        Initialize the Stock Prediction Loss.

        Args:
            delta (float): Weighting factor between L_rank and L_move.
            lambda_reg (float): Regularization coefficient for L2 penalty.
        """
        super(StockPredictionLoss, self).__init__()
        self.delta = delta
        self.lambda_reg = lambda_reg

    def forward(self, tau_F, e1, e2, b1, b2, y_return, y_move, mask=None):
        """
        Compute the loss.

        Args:
            tau_F (torch.Tensor) = Output from FinalLayer
            e1 (torch.Tensor): Task-specific hidden vector for return ratio (shape: [d]).
            e2 (torch.Tensor): Task-specific hidden vector for movement (shape: [d]).
            b1 (torch.Tensor): Bias term for return ratio (scalar).
            b2 (torch.Tensor): Bias term for movement (scalar).
            y_return (torch.Tensor): True return ratios (shape: [batch_size]).
            y_move (torch.Tensor): True binary movement labels (shape: [batch_size]).
            mask (torch.Tensor): Optional mask for valid pairs in L_rank (shape: [batch_size, batch_size]).
        
        Returns:
            torch.Tensor: Total loss (scalar).
        """
        # 1. Compute Predictions
        y_return_pred = torch.matmul(tau_F, e1) + b1  # Shape: [batch_size]
        y_move_pred = torch.sigmoid(torch.matmul(tau_F, e2) + b2)  # Shape: [batch_size]
        
        # 2. Compute Pairwise Ranking Loss (L_rank)
        pairwise_diff_pred = y_return_pred.unsqueeze(1) - y_return_pred.unsqueeze(0)  # Shape: [batch_size, batch_size]
        pairwise_diff_true = y_return.unsqueeze(1) - y_return.unsqueeze(0)  # Shape: [batch_size, batch_size]

        # Mask invalid pairs (optional)
        if mask is not None:
            pairwise_diff_pred = pairwise_diff_pred * mask
            pairwise_diff_true = pairwise_diff_true * mask

        # Max(0, -ˆΔ × Δ)
        l_rank = F.relu(-pairwise_diff_pred * pairwise_diff_true).mean()

        # 3. Compute Binary Cross-Entropy Loss for Movement (L_move)
        l_move = F.binary_cross_entropy(y_move_pred, y_move)

        # 4. L2 Regularization
        l2_reg = torch.sum(e1**2) + torch.sum(e2**2) + b1**2 + b2**2

        # 5. Total Loss
        total_loss = (1 - self.delta) * l_rank + self.delta * l_move + self.lambda_reg * l2_reg

        return total_loss
    
    # Main
import torch
import os

raw_dict = dict()
file_path=r'./raw_embeddings'
for file in os.listdir(file_path):
    data = torch.load(file_path+f"/{file}", weights_only=True)
    raw_dict[file[:-23]] = data

# i = 1
# for key in raw_dict.keys():
#     print(raw_dict[key].shape)
#     i += 1
#     if i > 5:
#         break

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.gru = SequenceEncoder(15,64)
        self.g_gat = GAT(64,64)
        self.t_attn = AttentionBlockLTSL(64,64,5)
        self.t_gat = SectorEmbeddingGenerator()
        self.final = FinalLayer()
    
    def forward(self, raw_dict, name_matrix):

        hashmapAi = dict()
        for key, data in raw_dict.items():
            hashmapAi[key] = self.gru(data)

        # adj_mat_intra_sector = name_matrix
        hashmapGi = dict()
        for name in name_matrix.keys():
            ai_sq = hashmapAi[name]
            ai_sn = []
            for sector_neighbour in name_matrix:
                ai_sn.append(hashmapAi[sector_neighbour])
            hashmapGi[name] = self.g_gat(ai_sq,ai_sn)
        return hashmapGi

class BatchBaseModel:
    def __init__(self, batch_size=10):
        self.base_model = BaseModel()
        self.batch_size = batch_size

    def process_embeddings(self, raw_dict, name_matrix):
        # Split raw_dict into batches
        stock_names = list(raw_dict.keys())
        batch_results = {}

        for i in range(0, len(stock_names), self.batch_size):
            batch_names = stock_names[i:i+self.batch_size]
            batch_data = {name: raw_dict[name] for name in batch_names}
            
            # Create a subset of name_matrix for this batch
            batch_name_matrix = {
                name: name_matrix.get(name, [])
                for name in batch_names
            }

            # Process batch
            batch_output = self.base_model(batch_data, batch_name_matrix)
            
            # Accumulate results
            batch_results.update(batch_output)

        return batch_results

batch_processor = BatchBaseModel(batch_size=10)
results = batch_processor.process_embeddings(raw_dict, name_matrix)

print(len(results.keys()))


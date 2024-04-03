import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


import torch
import torch.nn as nn

class TabPFN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[2**7, 2**7, 2**7, 2**7, 2**7, 2**7, 2**7], dropout=0.2):
        super(TabPFN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        torch.nn.init.xavier_normal_(self.input_layer.weight, gain=1.0)
        #self.input_dropout = nn.Dropout(dropout)  
        
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
                                            for i in range(len(hidden_sizes)-1)])
        self.hidden_bns = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes[1:]])
        self.hidden_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden_sizes)-1)])  
        for layer in self.hidden_layers:
            torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.output_bn = nn.BatchNorm1d(output_size)
        torch.nn.init.xavier_normal_(self.output_layer.weight, gain=1.0)
        #self.output_dropout = nn.Dropout(dropout)  
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.input_bn(self.input_layer(x)))
        #x = self.input_dropout(x)  
        for layer, bn, dropout in zip(self.hidden_layers, self.hidden_bns, self.hidden_dropouts):
            x = self.relu(bn(layer(x)))
            #x = dropout(x)  
        x = self.output_layer(x)
        x = self.output_bn(x)
        #x = self.output_dropout(x)  
        return x



VelocityEvaluationNetwork = TabPFN
SpeedIndexNetwork = TabPFN
    

# class Config(object): #各種設定
#     def __init__(self):
#         self.hidden_size = 2**9 #エンコーダの隠れ層の次元
#         self.intermediate_size = 2**10 #フィードフォワード層の中間のニューロンの数
#         self.hidden_dropout_prob = 0.1 #ドロップアウトの割合（確率）

#         self.num_attention_heads = 2**3 #アテンションヘッドの数　bertと同じ数に設定

#         self.max_position_embeddings = 30 #入力トークン数の最大値
#         self.num_hidden_layers = 12 #encoderレイヤーの数

# class AttentionHead(nn.Module): #アテンションヘッド
#     def __init__(self,
#                  embed_dim, #埋め込み次元
#                  head_dim): #アテンションヘッドの出力の次元
#         super().__init__()
#         self.q = nn.Linear(embed_dim, head_dim) #クエリ
#         self.k = nn.Linear(embed_dim, head_dim) #キー
#         self.v = nn.Linear(embed_dim, head_dim) #バリュー
        
#     def scaled_dot_produce_attention(self, query, key, value): #スケール化ドットアテンションの計算
#         dim_k = query.size(-1) #埋め込みベクトルサイズ
#         scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k) #クエリとキーの行列積をクエリのサイズでスケーリング
#         weights = F.softmax(scores, dim=-1) #ソフトマックスに入力
#         return torch.bmm(weights, value) #バリューに重みを乗じる

#     def forward(self,
#                 hidden_state): #クエリ、キー、バリューを入力してセルフアテンションの出力を得る
#         attn_outputs = self.scaled_dot_produce_attention(
#             self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        
#         return attn_outputs
    

# class MultiHeadAttention(nn.Module): #マルチヘッドアテンション
#     def __init__(self,
#                  config):
#         super().__init__()
#         embed_dim = config.hidden_size
#         num_heads = config.num_attention_heads
#         head_dim = embed_dim // num_heads

#         self.heads = nn.ModuleList(
#             [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]) #アテンションヘッドを設定
        
#         self.output_linear = nn.Linear(embed_dim, embed_dim) #全結合層

#     def forward(self,
#                 hidden_state):
#         x = torch.cat([h(hidden_state) for h in self.heads], dim=-1) #各アテンションヘッドの出力を連結
#         x = self.output_linear(x)
#         return x
    

# class FeedForward(nn.Module): #フィードフォワード層、単純な全結合だが各encoderに固有に設定する
#     def __init__(self, 
#                  config):
#         super().__init__()
#         self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
#         self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, x):
#         x = self.linear_1(x)
#         x = self.gelu(x)
#         x = self.linear_2(x)
#         x = self.dropout(x)

#         return x
    
# class TransformerEncoderLayer(nn.Module): #encoderレイヤー
#     def __init__(self,
#                  config):
#         super().__init__()
#         self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
#         self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
#         self.attention = MultiHeadAttention(config) #マルチヘッドアテンション
#         self.feed_forward = FeedForward(config) #フィードフォワード

#     def forward(self, x):
#         hidden_state = self.layer_norm_1(x)
#         x = x + self.attention(hidden_state) #アテンション重みを足す
#         x = x + self.feed_forward(self.layer_norm_2(x)) #フィードフォワード
#         return x 
    
# class TransformerEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.embeddings = Embeddings(config) #埋め込み
#         self.layers = nn.ModuleList([TransformerEncoderLayer(config) 
#                                      for _ in range(config.num_hidden_layers)]) #複数のencoderレイヤーを持つ
        
#     def forward(self, x):
#         x = self.embeddings(x)
#         for layer in self.layers: #encoderに連続で通す
#             x = layer(x)

#         return x
    

# class Embeddings(nn.Module):
#     def __init__(self, config):
#         super(Embeddings, self).__init__()
#         self.token_embedding_layer = nn.Linear(1, config.hidden_size)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings,
#                                                 config.hidden_size) #位置埋め込み
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout()
        

#     def forward(self, input_):
#         x = input_.unsqueeze(-1)
#         token_embeddings = self.token_embedding_layer(x)

#         #input_ = input_.long()
#         seq_length = input_.size(1) #入力の長さ
#         position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0) #seq_lengthの連番
#         position_ids = position_ids.to(input_.device)
#         position_embeddings = self.position_embeddings(position_ids)

#         embeddings = token_embeddings + position_embeddings #トークン埋め込みと位置埋め込みの和
#         embeddings = self.layer_norm(embeddings)
#         embeddings = self.dropout(embeddings)

#         return embeddings
    
    
# class TransformerRegressor(nn.Module): #分類ヘッドをつける
#     def __init__(self, input_dim, output_dim):
#         super().__init__()

#         config = Config()
#         config.max_position_embeddings = input_dim

#         self.encoder = TransformerEncoder(config) #transformerエンコーダ
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.regressor = nn.Linear(config.hidden_size, output_dim) #単純な全結合

#     def forward(self, x):
#         x = self.encoder(x)[:, 0, :]
#         x = self.dropout(x)
#         x = self.regressor(x)
#         return x




# class TabPFN(nn.Module):
#     def __init__(self, input_size, output_size, hidden_sizes=[2**7, 2**8, 2**9, 2**10, 2**9, 2**8, 2**7]):
#         super(TabPFN, self).__init__()
#         self.input_layer = nn.Linear(input_size, hidden_sizes[0])
#         self.input_bn = nn.BatchNorm1d(hidden_sizes[0])  # BatchNormを追加
#         torch.nn.init.xavier_normal_(self.input_layer.weight, gain=1.0)

#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) 
#                                             for i in range(len(hidden_sizes)-1)])
#         self.hidden_bns = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes[1:]])  # 各隠れ層にBatchNormを追加
#         for layer in self.hidden_layers:
#             torch.nn.init.xavier_normal_(layer.weight, gain=1.0)

#         self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
#         self.output_bn = nn.BatchNorm1d(output_size)  # 出力層にBatchNormを追加
#         torch.nn.init.xavier_normal_(self.output_layer.weight, gain=1.0)
        
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.relu(self.input_bn(self.input_layer(x)))  # BatchNormを適用
#         for layer, bn in zip(self.hidden_layers, self.hidden_bns):
#             x = self.relu(bn(layer(x)))  # BatchNormを適用
#         x = self.output_layer(x)
#         x = self.output_bn(x)  # BatchNormを適用
#         return x

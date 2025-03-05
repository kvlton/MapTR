from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    d = torch.tensor(q.shape[-1], dtype=torch.float32)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d)
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_p)
    output = torch.matmul(attention_weights, v)
    return output

class SubgraphNet_Layer(nn.Module):
    def __init__(self, input_channels=256, hidden_channels=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        x_max = torch.max(x, -2)[0]
        repeat_dims = [1] * len(x.shape)
        repeat_dims[-2] = x.shape[-2]
        x_max = x_max.unsqueeze(-2).repeat(repeat_dims)
        x = torch.cat([x, x_max], dim=-1)
        return x

class SubgraphNet(nn.Module):
    def __init__(self, input_channels=256, hidden_channels=128):
        super().__init__()
        self.sublayer1 = SubgraphNet_Layer(input_channels, hidden_channels)
        self.sublayer2 = SubgraphNet_Layer(hidden_channels * 2, hidden_channels)
        self.sublayer3 = SubgraphNet_Layer(hidden_channels * 2, hidden_channels)

    def forward(self, x):
        x = self.sublayer1(x)
        x = self.sublayer2(x)
        x = self.sublayer3(x)
        x_max = torch.max(x, -2)[0]
        return x_max


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x):
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        m = scaled_dot_product_attention(q, k, v)
        m = m.transpose(1, 2).flatten(start_dim=-2)
        m = self.out_proj(m)
        return x + self.ffn(torch.cat([x, m], -1))


class CrossAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.heads = num_heads
        self.to_qk = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x0, x1):
        qk0 = self.to_qk(x0)
        qk1 = self.to_qk(x1)
        v0 = self.to_v(x0)
        v1 = self.to_v(x1)
        qk0 = qk0.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        qk1 = qk1.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v0 = v0.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v1 = v1.unflatten(-1, (self.heads, -1)).transpose(1, 2)

        m0 = scaled_dot_product_attention(qk0, qk1, v1)
        m1 = scaled_dot_product_attention(qk1, qk0, v0)
        m0 = m0.transpose(1, 2).flatten(start_dim=-2)
        m1 = m1.transpose(1, 2).flatten(start_dim=-2)
        m0 = self.to_out(m0)
        m1 = self.to_out(m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.cross_attn = CrossAttention(embed_dim, num_heads)

    def forward(self, feature0, feature1):
        feature0 = self.self_attn(feature0)
        feature1 = self.self_attn(feature1)
        return self.cross_attn(feature0, feature1)


class HdmapMatcher(nn.Module):
    def __init__(self, num_layers=3, num_heads=4):
        super().__init__()
        self.input_project= nn.Linear(5, 128)
        self.input_embedding = nn.Linear(1, 128)
        # self.vector_net = SubgraphNet(6, 128)
        self.transformers = nn.ModuleList(
            [TransformerLayer(256, num_heads) for _ in range(num_layers)]
        )
        self.output_net = SubgraphNet(256, 128)
        self.reg_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3),
        )
    
    def forward(self, perception_features, hdmap_features):
        bs, n, p, d = perception_features.shape
        perception_features = perception_features.view(bs, n*p, d)
        bs, n, p, d = hdmap_features.shape
        hdmap_features = hdmap_features.view(bs, n*p, d)

        feature0 = torch.cat([self.input_project(perception_features[...,0:5]), 
                             self.input_embedding(perception_features[...,5:6])], dim=-1)
        feature1 = torch.cat([self.input_project(hdmap_features[...,0:5]), 
                             self.input_embedding(hdmap_features[...,5:6])], dim=-1)
        
        # feature0 = self.vector_net(perception_features)
        # feature1 = self.vector_net(hdmap_features)
        for transformer in self.transformers:
            feature0, feature1 = transformer(feature0, feature1)

        feature0 = self.output_net(feature0)
        feature1 = self.output_net(feature1)
        output_features = torch.cat((feature0, feature1), dim=-1)
        output = self.reg_branch(output_features)
        return output
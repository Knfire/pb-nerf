import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.networks.make_network import make_viewdir_embedder, make_residual, make_part_color_network, make_part_embedder
from lib.networks.embedders.part_base_embedder import Embedder as HashEmbedder
from lib.networks.embedders.freq_embedder import Embedder as FreqEmbedder

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout for attention weights
        """
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by the number of heads."

        # Linear transformations for generating Query, Key, and Value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query_input, key_input, value_input, mask=None):
        """
        query_input: Query sequence (B, L_query, embed_dim)
        key_input: Key sequence (B, L_key, embed_dim)
        value_input: Value sequence (B, L_value, embed_dim)
        mask: Mask (B, L_query, L_key)
        """
        B, L_query, _ = query_input.shape
        _, L_key, _ = key_input.shape

        # Generate query, key, and value
        q = self.query(query_input).view(B, L_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key_input).view(B, L_key, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value_input).view(B, L_key, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention weight calculation: q @ k^T / sqrt(d_k)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and dropout to the attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, L_query, self.embed_dim)

        # Final output after linear projection
        attn_output = self.proj_drop(self.out_proj(attn_output))
        return attn_output, attn_weights


# Assuming we have two features feat_a and feat_b with different channel dimensions, with shapes (N, C1) and (N, C2)
class FeatureFusionWithCrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_dim, num_heads, dropout=0.1):
        super(FeatureFusionWithCrossAttention, self).__init__()

        # Project the input features to the same dimension
        self.proj_a = nn.Linear(input_dim_a, output_dim)
        self.proj_b = nn.Linear(input_dim_b, output_dim)

        # Cross-attention mechanism
        self.cross_attention = CrossAttention(embed_dim=output_dim, num_heads=num_heads, dropout=dropout)

        # Final linear projection to map the output to (N, 16)
        self.final_proj = nn.Linear(output_dim * 2, 16)

    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: Feature A, shape (N, C1)
            feat_b: Feature B, shape (N, C2)
        Returns:
            Cross-attention output features, shape (N, 16)
        """
        # Add a dimension to (N, C1) and (N, C2) to match the input shape for multi-head attention
        feat_a = feat_a.unsqueeze(1)  # (N, 1, C1)
        feat_b = feat_b.unsqueeze(1)  # (N, 1, C2)

        # Project the features to the same dimension
        feat_a = self.proj_a(feat_a)  # Map feat_a to (N, 1, output_dim)
        feat_b = self.proj_b(feat_b)  # Map feat_b to (N, 1, output_dim)

        # Use feat_a as the query, feat_b as the key and value
        output_a, attn_weights_a = self.cross_attention(feat_a, feat_b, feat_b)

        # Use feat_b as the query, feat_a as the key and value
        output_b, attn_weights_b = self.cross_attention(feat_b, feat_a, feat_a)

        # Concatenate the two cross-attention outputs
        fused_output = torch.cat([output_a, output_b], dim=2)  # (N, 1, output_dim * 2)

        # Project the fused output to (N, 16)
        fused_output = self.final_proj(fused_output.squeeze(1))  # (N, 16)

        return fused_output


class MLP(nn.Module):
    def __init__(self, indim=16, outdim=3, d_hidden=64, n_layers=2):
        super(MLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.linears = nn.ModuleList([nn.Linear(indim, d_hidden)] + [nn.Linear(d_hidden, d_hidden) for i in range(n_layers - 1)] + [nn.Linear(d_hidden, outdim)])
        self.actvn = nn.Softplus()

    def forward(self, input):
        net = input
        for i, l in enumerate(self.linears[:-1]):
            net = self.actvn(l(net))
        net = self.linears[-1](net)
        return net

ColorNetwork = MLP


class Network(nn.Module):
    def __init__(self, partname, pid):
        super().__init__()
        self.pid = pid
        self.partname = partname

        self.embedder: HashEmbedder = make_part_embedder(cfg, partname, pid)
        self.embedder_dir: FreqEmbedder = make_viewdir_embedder(cfg)
        self.occ = MLP(self.embedder.out_dim, 1 + cfg.geo_feature_dim, **cfg.network.occ)
        indim_rgb = self.embedder.out_dim + self.embedder_dir.out_dim + cfg.geo_feature_dim + 16
        # indim_rgb = self.embedder.out_dim + self.embedder_dir.out_dim + cfg.geo_feature_dim + cfg.latent_code_dim
        self.rgb_latent = nn.Parameter(torch.zeros(cfg.num_latent_code, cfg.latent_code_dim))
        nn.init.kaiming_normal_(self.rgb_latent)
        self.rgb = make_part_color_network(cfg, partname, indim=indim_rgb)


    def forward(self, tpts: torch.Tensor, viewdir: torch.Tensor, dists: torch.Tensor, batch, feat):
    # def forward(self, tpts: torch.Tensor, viewdir: torch.Tensor, dists: torch.Tensor, batch):
        # tpts: N, 3
        # viewdir: N, 3
        N, D = tpts.shape
        C, L = self.rgb_latent.shape
        embedded = self.embedder(tpts, batch)  # embedding
        hidden: torch.Tensor = self.occ(embedded)  # networking
        occ = 1 - torch.exp(-self.occ.actvn(hidden[..., :1]))  # activation
        feature = hidden[..., 1:]

        embedded_dir = self.embedder_dir(viewdir, batch)  # embedding
        latent_code = self.rgb_latent.gather(dim=0, index=batch['latent_index'].expand(N, L))  # NOTE: ignoring batch dimension

        # feature_fusion_layer = FeatureFusionWithCrossAttention(input_dim_a=cfg.latent_code_dim, input_dim_b=getattr(cfg.partnet, self.partname).dim,
        #                                                    output_dim=64, num_heads=1, dropout=0.1).cuda()

        feature_fusion_layer = FeatureFusionWithCrossAttention(input_dim_a=cfg.latent_code_dim, input_dim_b=60,
                                                           output_dim=64, num_heads=1, dropout=0.1).cuda()
        fused_features = feature_fusion_layer(latent_code, feat)

        # input = torch.cat([embedded, feature, fused_features], dim=-1)
        input = torch.cat([embedded, embedded_dir, feature, fused_features], dim=-1)
        # input = torch.cat([embedded, embedded_dir, feature, latent_code], dim=-1)
        rgb: torch.Tensor = self.rgb(input)  # networking
        rgb = rgb.sigmoid()  # activation

        raw = torch.cat([rgb, occ], dim=-1)
        ret = {'raw': raw, 'occ': occ}

        return ret



import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    """Configuration for BrainNet Model"""

    # feature_dim = 1632  # ROI feature dimension
    # num_rois = 400  # Number of ROIs
    # cluster_shape = (45, 54, 45)  # 3D cluster index shape
    # hidden_dim = 16  # Balanced size (was 16 - too small, 64 - might overfit)
    # num_heads = 4  # Number of attention heads
    # num_layers = 1  # Balanced (was 1 - too small, 3 - might overfit)
    # intermediate_dim = hidden_dim * 4
    # dropout = 0.5
    # block_size = 9  # Spatial block size (K x K x K)
    # block_stride = 5  # Stride for block pooling
    # output_dim = 2  # Binary classification

    def __init__(self,
                 feature_dim=1632,
                 num_rois=400,  # Number of ROIs
                 cluster_shape = (45, 54, 45),
                 hidden_dim=16,
                 num_heads = 4,  # Number of attention heads
                 num_layers = 1,  # Balanced (was 1 - too small, 3 - might overfit)
                 intermediate_dim = 64,
                 dropout = 0.5,
                 block_size = 9,  # Spatial block size (K x K x K)
                 block_stride = 5,  # Stride for block pooling
                 output_dim = 2  # Binary classification
                 ):

        self.feature_dim = feature_dim
        self.num_rois = num_rois
        self.cluster_shape = cluster_shape
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_dim = self.hidden_dim * 4
        self.dropout = dropout
        self.block_size = block_size
        self.block_stride = block_stride
        self.output_dim = output_dim




class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        return output



class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        return x




class BrainNet(nn.Module):
    """Atlas-free Brain Network Transformer (FIXED VERSION)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Step 1: ROI Connectivity Projection (FIXED: BatchNorm1d → LayerNorm)
        self.roi_projection = nn.Sequential(
            nn.Linear(config.feature_dim, 64),  # Intermediate dimension
            nn.LayerNorm(64),  # FIX: LayerNorm works on last dimension [B, num_rois, 64]
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, config.hidden_dim)
        )

        # # Step 1: ROI Connectivity Projection (FIXED: BatchNorm1d → LayerNorm)
        # self.roi_projection = nn.Sequential(
        #     nn.Linear(config.feature_dim, config.hidden_dim),  # Intermediate dimension
        #     nn.LayerNorm(64),  # FIX: LayerNorm works on last dimension [B, num_rois, 64]
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        # )
        
        self.Q_pooling = nn.AvgPool3d(
            kernel_size=self.config.block_size,
            stride=self.config.block_stride,
            padding=0
        )

        # 2. DYNAMIC CALCULATION of Token Count
        # Extract dimensions from config
        D_in, H_in, W_in = config.cluster_shape
        K = config.block_size
        S = config.block_stride
        
        # Calculate the number of nodes after 3D pooling and sum-pooling
        self.D_out = (D_in - K) // S + 1
        self.H_out = (H_in - K) // S + 1
        self.W_out = (W_in - K) // S + 1
        
        self.num_tokens = self.D_out * self.H_out * self.W_out
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, config.hidden_dim))
        
        # Step 4: Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.num_heads,
                config.intermediate_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Step 5: Global Readout & Classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
    
    def _construct_3d_brain_map(self, q, c_mat):

        batch_size, num_rois, hidden_dim = q.shape
        D, H, W = c_mat.shape[1], c_mat.shape[2], c_mat.shape[3]
        
        # Initialize output tensor (background will remain zero)
        Q = torch.zeros(batch_size, hidden_dim, D, H, W, 
                        device=q.device, dtype=q.dtype)
        
        # Process each sample in the batch
        for b in range(batch_size):
            
            cluster_indices = c_mat[b]  # Contains: 0 (background), 1-400 (ROI indices)
            
            roi_features = q[b]
            
            cluster_indices_flat = cluster_indices.flatten()  # [D*H*W]
            
            non_bg_mask = (cluster_indices_flat > 0) & (cluster_indices_flat <= num_rois)
            
            roi_indices = (cluster_indices_flat[non_bg_mask] - 1).long()
            
            voxel_features = torch.zeros(D * H * W, hidden_dim, 
                                        device=q.device, dtype=q.dtype)
            voxel_features[non_bg_mask] = roi_features[roi_indices]  # [num_non_bg, hidden_dim]
            
            # Transpose to get [hidden_dim, D, H, W] format
            Q[b] = voxel_features.view(D, H, W, hidden_dim).permute(3, 0, 1, 2)
        
        return Q

    
    def forward(self, f_mat, c_mat):
        batch_size = f_mat.size(0)
        
        q = self.roi_projection(f_mat)  # [B, num_rois, hidden_dim]
        
        Q = self._construct_3d_brain_map(q, c_mat)  # [B, hidden_dim, D, H, W]
        
        Q_pooled = self.Q_pooling(Q)

        D_out, H_out, W_out = Q_pooled.shape[2], Q_pooled.shape[3], Q_pooled.shape[4]
        num_tokens = D_out * H_out * W_out
        
        if num_tokens != self.num_tokens:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, num_tokens, self.config.hidden_dim, device=Q_pooled.device)
            ).to(Q_pooled.device)
            self.num_tokens = num_tokens
        
        tokens = Q_pooled.view(batch_size, self.config.hidden_dim, num_tokens)
        tokens = tokens.transpose(1, 2)  # [B, num_tokens, hidden_dim]
        
        x = tokens + self.pos_embedding
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        x = x.transpose(1, 2)  # [B, hidden_dim, num_tokens]
        x = self.global_pool(x).squeeze(-1)  # [B, hidden_dim]
        logits = self.classifier(x)  # [B, output_dim]
        
        return logits






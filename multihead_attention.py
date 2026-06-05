import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # Precompute rotary embedding angles
        self.max_seq_len = max_seq_len
        self._precompute_rope_angles()

    def _precompute_rope_angles(self):
        """Precompute the rotary embedding angles for positions up to max_seq_len."""
        theta = 10000.0 ** (-2.0 * (torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k))
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * theta[None, :]  # Shape: [max_seq_len, d_k//2]
        self.register_buffer("cos_cache", torch.cos(angles))
        self.register_buffer("sin_cache", torch.sin(angles))

    def _apply_rotary_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary positional embeddings to the input tensor."""
        batch_size, num_heads, seq_len_x, d_k = x.size()
        assert seq_len_x <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # Reshape x to separate dimensions for rotation
        x_ = x.view(batch_size, num_heads, seq_len_x, d_k // 2, 2)
        x1, x2 = x_[..., 0], x_[..., 1]  # Split into pairs

        # Get cos and sin for the sequence length
        cos = self.cos_cache[:seq_len_x].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k//2]
        sin = self.sin_cache[:seq_len_x].unsqueeze(0).unsqueeze(0)

        # Apply rotation: x1' = x1 * cos - x2 * sin, x2' = x1 * sin + x2 * cos
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Combine rotated pairs back
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1).view(batch_size, num_heads, seq_len_x, d_k)
        return x_rot

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Linear transformations and reshape
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply rotary embeddings to query and key
        q = self._apply_rotary_emb(q, seq_len)
        k = self._apply_rotary_emb(k, seq_len)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.w_o(context)
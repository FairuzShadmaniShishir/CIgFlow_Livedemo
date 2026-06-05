import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Initialize relative positional embeddings
        self.rel_pe = nn.Parameter(torch.zeros(2 * max_len - 1, d_model))
        nn.init.normal_(self.rel_pe, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model] (after transpose in GPT2LikeDecoder)

        Returns:
            Tensor of shape [seq_len, batch_size, d_model] with relative positional encodings added
        """
        seq_len = x.size(0)  # seq_len is first dimension after transpose
        batch_size = x.size(1)

        # Compute relative position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, 1]
        rel_pos = positions - torch.arange(seq_len, device=x.device).unsqueeze(0)  # [seq_len, seq_len]
        rel_pos = rel_pos.clamp(-self.max_len + 1, self.max_len - 1) + self.max_len - 1  # [seq_len, seq_len]

        # Index into rel_pe to get positional embeddings
        rel_pe = self.rel_pe[rel_pos]  # [seq_len, seq_len, d_model]

        # Average or select positional embeddings to match [seq_len, batch_size, d_model]
        # For simplicity, take the first seq_len embeddings (or use attention-specific logic)
        rel_pe = rel_pe[:, 0, :].unsqueeze(1).expand(-1, batch_size, -1)  # [seq_len, batch_size, d_model]

        return x + rel_pe.to(x.device)
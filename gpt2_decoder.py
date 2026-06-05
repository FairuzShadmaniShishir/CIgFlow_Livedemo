import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock
from positional_encoding import PositionalEncoding


class GPT2LikeDecoder(nn.Module):
    """GPT-2 style decoder for nanobody generation"""

    def __init__(self, latent_dim: int, vocab_size: int, d_model: int = 256,
                 num_heads: int = 8, num_layers: int = 4, d_ff: int = 1024,
                 max_seq_len: int = 100, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.latent_dim = latent_dim

        # Latent to initial token embedding
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (GPT-2 style)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len: int, device: torch.device):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, latent: torch.Tensor, target_tokens: torch.Tensor = None):
        """
        Args:
            latent: (batch_size, latent_dim) - latent representation from flow model
            target_tokens: (batch_size, seq_len) - target tokens for training
        """
        batch_size = latent.size(0)
        device = latent.device

        if target_tokens is not None:
            # Training mode - use teacher forcing
            seq_len = target_tokens.size(1)

            # Get token embeddings
            token_emb = self.token_embedding(target_tokens)  # (batch_size, seq_len, d_model)

            # Add latent information to first token
            latent_emb = self.latent_proj(latent).unsqueeze(1)  # (batch_size, 1, d_model)
            token_emb[:, 0:1, :] = token_emb[:, 0:1, :] + latent_emb

        else:
            # Inference mode - start with latent
            seq_len = 1
            token_emb = self.latent_proj(latent).unsqueeze(1)  # (batch_size, 1, d_model)

        # Add positional encoding
        x = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, latent: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                 pad_token_id: int = 0):
        """Generate sequence autoregressively from latent"""
        self.eval()
        batch_size = latent.size(0)
        device = latent.device

        # Start with pad token
        generated = torch.full((batch_size, 1), pad_token_id, device=device, dtype=torch.long)

        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.forward(latent, generated)

            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated
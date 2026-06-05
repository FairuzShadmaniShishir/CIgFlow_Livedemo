import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        return x + residual


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb




class ConditionalFlowMatchingModel(nn.Module):
    """
    Conditional Flow Matching model with merged embedding and antigen conditioning.
    The antigen embedding can be masked for classifier-free guidance.
    """

    def __init__(
            self,
            merged_embed_dim: int,
            antigen_embed_dim: int,
            hidden_dim: int,
            seq_dim: int,
            num_layers: int = 4,
            dropout: float = 0.1,
            use_time_embedding: bool = True,
            cfg_dropout: float = 0.1,
    ):
        super().__init__()

        self.merged_embed_dim = merged_embed_dim
        self.antigen_embed_dim = antigen_embed_dim
        self.seq_dim = seq_dim
        self.use_time_embedding = use_time_embedding
        self.cfg_dropout = cfg_dropout

        # Time embedding
        if use_time_embedding:
            self.time_embed = SinusoidalPositionalEmbedding(hidden_dim // 4)
            input_dim = seq_dim + merged_embed_dim + antigen_embed_dim + hidden_dim // 4
        else:
            input_dim = seq_dim + merged_embed_dim + antigen_embed_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual layers
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection (velocity field)
        self.output_proj = nn.Linear(hidden_dim, seq_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            x_t: torch.Tensor,
            merged_emb: torch.Tensor,
            antigen_emb: torch.Tensor,
            t: torch.Tensor,
            mask_condition: bool = False
    ):
        """
        Forward pass predicting velocity field.

        Args:
            x_t: interpolated state [batch, seq_dim]
            merged_emb: merged embedding [batch, merged_embed_dim]
            antigen_emb: antigen embedding [batch, antigen_embed_dim]
            t: time tensor [batch, 1]
            mask_condition: if True, zero out antigen conditioning (for CFG)
        """
        assert x_t.size(-1) == self.seq_dim, f"Expected seq_dim {self.seq_dim}, got {x_t.size(-1)}"
        assert merged_emb.size(-1) == self.merged_embed_dim
        assert antigen_emb.size(-1) == self.antigen_embed_dim

        # Apply conditioning dropout for classifier-free guidance
        if mask_condition:
            antigen_emb = torch.zeros_like(antigen_emb)

        inputs = [x_t, merged_emb, antigen_emb]

        # Add time embedding
        if self.use_time_embedding:
            time_emb = self.time_embed(t)
            inputs.append(time_emb)

        x = torch.cat(inputs, dim=-1)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)

    def compute_cfm_loss(
            self,
            x_1: torch.Tensor,
            merged_emb: torch.Tensor,
            antigen_emb: torch.Tensor
    ):
        """
        Compute Conditional Flow Matching loss with optimal transport.

        Args:
            x_1: target antibody sequences [batch, seq_dim]
            merged_emb: merged embedding [batch, merged_embed_dim]
            antigen_emb: antigen embedding [batch, antigen_embed_dim]

        Returns:
            loss: CFM loss value
        """
        batch_size = x_1.size(0)
        device = x_1.device

        # Sample time uniformly from [0, 1]
        t = torch.rand(batch_size, 1, device=device)

        # Sample source from standard normal (prior distribution)
        x_0 = torch.randn_like(x_1)

        # Optimal transport conditional probability path
        x_t = t * x_1 + (1 - t) * x_0

        # True conditional velocity field
        u_t = x_1 - x_0

        # Random dropout of conditioning for classifier-free guidance training
        mask_condition = torch.rand(batch_size, device=device) < self.cfg_dropout

        # Predict velocity field
        v_t = []
        for i in range(batch_size):
            v = self.forward(
                x_t[i:i + 1],
                merged_emb[i:i + 1],
                antigen_emb[i:i + 1],
                t[i:i + 1],
                mask_condition=mask_condition[i].item()
            )
            v_t.append(v)
        v_t = torch.cat(v_t, dim=0)

        # CFM loss: match the conditional vector field
        loss = F.mse_loss(v_t, u_t)

        return loss

    def compute_vp_cfm_loss(
            self,
            x_1: torch.Tensor,
            merged_emb: torch.Tensor,
            antigen_emb: torch.Tensor
    ):
        """
        Variance-preserving CFM loss with better numerical stability.

        Uses cosine schedule: x_t = cos(πt/2) * x_1 + sin(πt/2) * x_0
        """
        batch_size = x_1.size(0)
        device = x_1.device

        t = torch.rand(batch_size, 1, device=device)
        x_0 = torch.randn_like(x_1)

        # Variance-preserving interpolation
        alpha_t = torch.cos(t * math.pi / 2)
        sigma_t = torch.sin(t * math.pi / 2)

        x_t = alpha_t * x_1 + sigma_t * x_0

        # Velocity includes derivative of noise schedule
        u_t = -(math.pi / 2) * (torch.sin(t * math.pi / 2) * x_1 -
                                torch.cos(t * math.pi / 2) * x_0)

        # Conditioning dropout
        mask_condition = torch.rand(batch_size, device=device) < self.cfg_dropout

        v_t = []
        for i in range(batch_size):
            v = self.forward(
                x_t[i:i + 1],
                merged_emb[i:i + 1],
                antigen_emb[i:i + 1],
                t[i:i + 1],
                mask_condition=mask_condition[i].item()
            )
            v_t.append(v)
        v_t = torch.cat(v_t, dim=0)

        loss = F.mse_loss(v_t, u_t)
        return loss


    @torch.no_grad()
    def euler_sample(
            self,
            merged_emb: torch.Tensor,
            antigen_emb: torch.Tensor,
            num_steps: int = 100,
            guidance_scale: float = 0.0
    ):
        """
        Euler integration sampling with optional classifier-free guidance.

        Args:
            merged_emb: merged embedding [batch, merged_embed_dim]
            antigen_emb: antigen embedding [batch, antigen_embed_dim]
            num_steps: number of integration steps
            guidance_scale: CFG scale (0.0 = no guidance, higher = stronger conditioning)
        """
        batch_size = merged_emb.size(0)
        device = merged_emb.device

        # Initialize from prior (standard normal)
        x_t = torch.randn(batch_size, self.seq_dim, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_i = torch.full((batch_size, 1), i / num_steps, device=device)

            if guidance_scale > 0.0:
                # Classifier-free guidance
                v_cond = self.forward(x_t, merged_emb, antigen_emb, t_i, mask_condition=False)
                v_uncond = self.forward(x_t, merged_emb, antigen_emb, t_i, mask_condition=True)
                v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                # Standard conditional sampling
                v_t = self.forward(x_t, merged_emb, antigen_emb, t_i, mask_condition=False)

            x_t = x_t + v_t * dt

        return x_t

    @torch.no_grad()
    def heun_sample(
            self,
            merged_emb: torch.Tensor,
            antigen_emb: torch.Tensor,
            num_steps: int = 50,
            guidance_scale: float = 0.0,
            noise_scale: float = 1.0
    ):
        """
        Heun's method (RK2) sampling with classifier-free guidance.
        Higher quality but twice as many model evaluations.
        """
        batch_size = merged_emb.size(0)
        device = merged_emb.device

        x_t = torch.randn(batch_size, self.seq_dim, device=device)
        dt = 1.0 / num_steps

        def get_velocity(x, t):
            if guidance_scale > 0.0:
                v_cond = self.forward(x, merged_emb, antigen_emb, t, mask_condition=False)
                v_uncond = self.forward(x, merged_emb, antigen_emb, t, mask_condition=True)
                return v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                return self.forward(x, merged_emb, antigen_emb, t, mask_condition=False)

        for i in range(num_steps):
            t_i = torch.full((batch_size, 1), i / num_steps, device=device)
            t_next = torch.full((batch_size, 1), (i + 1) / num_steps, device=device)

            # First velocity estimate
            v1 = get_velocity(x_t, t_i)

            # Predicted next state
            x_pred = x_t + v1 * dt

            # Second velocity estimate
            v2 = get_velocity(x_pred, t_next)

            # Average of velocities
            x_t = x_t + (v1 + v2) * dt / 2

        return x_t



    @torch.no_grad()
    def sample_with_trajectory(
            self,
            merged_emb,
            antigen_emb,
            num_steps=100,
            guidance_scale=0.0
    ):
        batch_size = merged_emb.size(0)
        device = merged_emb.device

        x_t = torch.randn(batch_size, self.seq_dim, device=device)

        dt = 1.0 / num_steps

        trajectory = []

        for i in range(num_steps):
            t_i = torch.full((batch_size, 1), i / num_steps, device=device)

            if guidance_scale > 0.0:
                v_cond = self.forward(x_t, merged_emb, antigen_emb, t_i, mask_condition=False)
                v_uncond = self.forward(x_t, merged_emb, antigen_emb, t_i, mask_condition=True)
                v_t = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_t = self.forward(x_t, merged_emb, antigen_emb, t_i, mask_condition=False)

            x_t = x_t + v_t * dt

            # 🔥 STORE TRAJECTORY
            trajectory.append(x_t.detach().cpu())

        return x_t, trajectory




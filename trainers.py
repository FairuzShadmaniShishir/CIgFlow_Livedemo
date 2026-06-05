import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlowMatchingTrainer:
    def __init__(self, model, optimizer, scheduler=None, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

        # EMA for model parameters
        self.ema_decay = 0.9999
        self.ema_params = {}
        for name, param in model.named_parameters():
            self.ema_params[name] = param.data.clone()

    def train_step(self, z, real_seq, embeddings, t):
        self.model.train()

        # Interpolation and vector field
        x_t = self.interpolate(z, real_seq, t)
        v_target = self.target_vector_field(z, real_seq, t)

        # Forward pass
        v_pred = self.model(x_t, embeddings, t)

        # Loss with regularization
        loss = F.mse_loss(v_pred, v_target)

        # Add L2 regularization
        l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
        loss = loss + 1e-5 * l2_reg

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update EMA
        self.update_ema()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()

    def update_ema(self):
        for name, param in self.model.named_parameters():
            self.ema_params[name] = self.ema_decay * self.ema_params[name] + (1 - self.ema_decay) * param.data

    def apply_ema(self):
        """Apply EMA weights for inference"""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.ema_params[name])

    @staticmethod
    def interpolate(x0, x1, t):
        return (1 - t) * x0 + t * x1

    @staticmethod
    def target_vector_field(x0, x1, t):
        return x1 - x0


class GPT2DecoderTrainer:
    def __init__(self, model, optimizer, scheduler=None, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

    def compute_perplexity(self, data_loader):
        """
        Compute perplexity of the GPT-2 decoder on a given dataset.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_latent, batch_target in data_loader:
                # Forward pass with teacher forcing
                logits = self.model(batch_latent, batch_target[:, :-1])  # Exclude last token for input

                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.contiguous().view(-1, logits.size(-1)),
                    batch_target[:, 1:].contiguous().view(-1),  # Exclude first token for target
                    reduction='sum'  # Sum the loss over all tokens
                )

                # Accumulate loss and count tokens
                total_loss += loss.item()
                total_tokens += batch_target[:, 1:].contiguous().view(-1).size(0)

        # Compute average negative log-likelihood
        avg_nll = total_loss / total_tokens

        # Compute perplexity
        perplexity = math.exp(avg_nll)

        return perplexity

    def train_step(self, latents, targets):
        self.model.train()

        # Forward pass with teacher forcing
        logits = self.model(latents, targets[:, :-1])  # Exclude last token for input

        # Calculate loss (predict next token)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            targets[:, 1:].contiguous().view(-1)  # Exclude first token for target
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()
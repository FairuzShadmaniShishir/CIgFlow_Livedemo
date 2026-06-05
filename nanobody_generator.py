import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.model_selection import train_test_split

#from flow_matching_model import RobustFlowMatchingModel, ConditionalFlowMatchingModel
from flow_matching_model import  ConditionalFlowMatchingModel
from gpt2_decoder import GPT2LikeDecoder
from trainers import FlowMatchingTrainer, GPT2DecoderTrainer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class NanobodyGenerator:
    def __init__(self, seq_dim=128, max_seq_len= 1024, hidden_dim=512):
        self.seq_dim = seq_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Load nanoBERT
        # self.tokenizer = AutoTokenizer.from_pretrained("NaturalAntibody/nanoBERT")
        # self.embedding_model = AutoModel.from_pretrained("NaturalAntibody/nanoBERT").to(device)

        model_name = "ollieturnbull/p-IgGen"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name).to(device)

        # self.tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
        # self.embedding_model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False).to(device)
        # Use nanoBERT's vocabulary
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Models will be initialized later
        self.flow_model = None
        self.decoder = None
        self.flow_trainer = None
        self.decoder_trainer = None

        # Encoder layers for generating latent representations
        self.encoder_layers = None

    def get_embedding(self, sequence: str):
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=False,
                                max_length=self.max_seq_len).to(device)

        # ✅ REMOVE token_type_ids (GPT-style models don't support it)
        inputs.pop("token_type_ids", None)

        # Move to device AFTER cleanup
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)

            #print(outputs.last_hidden_state.mean(dim=1).squeeze(0))
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    def sequence_to_tensor(self, seq: str):
        # Tokenize using BPE
        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=True
        )
        return inputs['input_ids'].squeeze(0).to(device)

    def tokens_to_sequence(self, tokens):
        # Convert token indices to sequence using tokenizer
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def prepare_dataset(self, antibody_sequences, antigen_sequences, batch_size=16):
        logger.info(f"Preparing dataset with {len(antibody_sequences)} antibody–antigen pairs")

        merged_embeddings = []  # Combined ab+ag embeddings
        antigen_embeddings = []  # Separate antigen embeddings
        targets = []

        for ab_seq, ag_seq in zip(antibody_sequences, antigen_sequences):
            try:
                # Compute embeddings separately
                ab_emb = self.get_embedding(ab_seq)  # shape: (seq_dim,)
                ag_emb = self.get_embedding(ag_seq)  # shape: (seq_dim,)

                # Concatenate embeddings → joint representation (merged embedding)
                combined_emb = torch.cat([ab_emb, ag_emb], dim=-1)  # shape: (2 * seq_dim,)

                # Create target (antibody sequence tensor)
                target = self.sequence_to_tensor(ab_seq)

                merged_embeddings.append(combined_emb)
                antigen_embeddings.append(ag_emb)  # Store antigen embedding separately
                targets.append(target)

            except Exception as e:
                logger.warning(f"Failed to process pair: {e}")
                continue

        merged_embeddings = torch.stack(merged_embeddings).to(device)
        antigen_embeddings = torch.stack(antigen_embeddings).to(device) # Stack all antigen embeddings
        targets = torch.stack(targets).to(device)

        # ⚡ Save precomputed embeddings to disk
        # torch.save({
        #     "merged": merged_embeddings,
        #     "antigen": antigen_embeddings,
        #     "targets": targets
        # }, "precomputed_dataset.pt")  # you can change filename

        # Create latent space representation
        X = torch.randn(len(merged_embeddings), self.seq_dim).to(device)

        # Debug: verify shapes match
        #print(f"X shape: {X.shape}")
        #print(f"merged_embeddings shape: {merged_embeddings.shape}")
        #print(f"antigen_embeddings shape: {antigen_embeddings.shape}")

        # Create data loaders - now using merged_embeddings and antigen_embeddings
        flow_dataset = TensorDataset(X, merged_embeddings, antigen_embeddings)
        decoder_dataset = TensorDataset(X, targets)

        flow_loader = DataLoader(flow_dataset, batch_size=batch_size, shuffle=True)
        decoder_loader = DataLoader(decoder_dataset, batch_size=batch_size, shuffle=True)

        return flow_loader, decoder_loader, merged_embeddings, antigen_embeddings, targets, X

    def initialize_models(self, merged_embed_dim: int, antigen_embed_dim: int):
        # Flow matching model
        self.flow_model = ConditionalFlowMatchingModel(
            merged_embed_dim=merged_embed_dim,
            antigen_embed_dim=antigen_embed_dim,
            hidden_dim=self.hidden_dim,
            seq_dim=self.seq_dim,
            num_layers=4,
            dropout=0.1,
            use_time_embedding=True,
            cfg_dropout=0.1  # 10% chance to mask antigen during training
        ).to(device)

        # GPT-2 like decoder
        self.decoder = GPT2LikeDecoder(
            latent_dim=self.seq_dim,
            vocab_size=self.vocab_size,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            max_seq_len=self.max_seq_len,
            dropout=0.1
        ).to(device)

        # Simple encoder layers to convert embeddings to latent space
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(merged_embed_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.seq_dim),
                nn.Tanh()
            )
        ]).to(device)

        # Optimizers
        self.flow_optimizer = optim.AdamW(self.flow_model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=1e-4, weight_decay=1e-5)

        # Schedulers
        self.flow_scheduler = CosineAnnealingLR(self.flow_optimizer, T_max=1000)
        self.decoder_scheduler = CosineAnnealingLR(self.decoder_optimizer, T_max=10000)

        # Trainers
        self.flow_trainer = FlowMatchingTrainer(self.flow_model, self.flow_optimizer, self.flow_scheduler)
        self.decoder_trainer = GPT2DecoderTrainer(self.decoder, self.decoder_optimizer, self.decoder_scheduler)

        logger.info(f"Flow model parameters: {sum(p.numel() for p in self.flow_model.parameters()):,}")
        logger.info(f"Decoder parameters: {sum(p.numel() for p in self.decoder.parameters()):,}")

    # def train_flow_model(self, flow_loader, embeddings, X, epochs=1000):
    #     logger.info("Training flow matching model...")
    #
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         num_batches = 0
    #
    #         for batch_x, batch_emb in flow_loader:
    #             # Sample time and noise
    #             t = torch.rand((batch_x.size(0), 1), device=device)
    #             z = torch.randn_like(batch_x)
    #
    #             loss = self.flow_trainer.train_step(z, batch_x, batch_emb, t)
    #             total_loss += loss
    #             num_batches += 1
    #
    #         avg_loss = total_loss / num_batches
    #
    #         if epoch % 100 == 0:
    #             logger.info(f"Flow Epoch {epoch}: Loss = {avg_loss:.6f}")
    #
    #     # Apply EMA weights
    #     self.flow_trainer.apply_ema()
    #     logger.info("Flow model training completed!")

    def train_flow_model(self, flow_loader, embeddings, X, epochs=1000, use_vp=False):
        """
        Train the conditional flow matching model.

        Args:
            flow_loader: DataLoader providing (batch_x, batch_merged_emb, batch_antigen_emb)
            embeddings: Not used (kept for compatibility)
            X: Not used (kept for compatibility)
            epochs: Number of training epochs
            use_vp: If True, use variance-preserving CFM loss
        """
        logger.info("Training flow matching model...")

        # Put model in training mode
        self.flow_model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_data in flow_loader:
                # Unpack batch - expecting (x_1, merged_emb, antigen_emb)
                if len(batch_data) == 3:
                    batch_x, batch_merged_emb, batch_antigen_emb = batch_data
                elif len(batch_data) == 2:
                    # If only 2 items, assume second is merged embedding
                    batch_x, batch_merged_emb = batch_data
                    batch_antigen_emb = None
                else:
                    raise ValueError(f"Expected 2 or 3 items from dataloader, got {len(batch_data)}")

                # Move to device
                batch_x = batch_x.to(device)
                batch_merged_emb = batch_merged_emb.to(device)
                if batch_antigen_emb is not None:
                    batch_antigen_emb = batch_antigen_emb.to(device)
                else:
                    # Create dummy antigen embedding if not provided
                    batch_antigen_emb = torch.zeros(
                        batch_x.size(0),
                        self.flow_model.antigen_embed_dim,
                        device=device
                    )

                # Compute loss using the model's built-in loss function
                self.flow_optimizer.zero_grad()

                if use_vp:
                    loss = self.flow_model.compute_vp_cfm_loss(
                        batch_x,
                        batch_merged_emb,
                        batch_antigen_emb
                    )
                else:
                    loss = self.flow_model.compute_cfm_loss(
                        batch_x,
                        batch_merged_emb,
                        batch_antigen_emb
                    )

                loss.backward()

                # Optional: gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)

                self.flow_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            if epoch % 100 == 0:
                logger.info(f"Flow Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")

            # Optional: learning rate scheduling
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

        # Apply EMA weights if available
        if hasattr(self, 'flow_trainer') and hasattr(self.flow_trainer, 'apply_ema'):
            self.flow_trainer.apply_ema()

        logger.info("Flow model training completed!")

    def train_decoder(self, decoder_loader, val_loader=None, epochs=5000):
        logger.info("Training GPT-2 decoder...")

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_latent, batch_target in decoder_loader:
                loss = self.decoder_trainer.train_step(batch_latent, batch_target)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches

            if epoch % 2 == 0:
                # Compute perplexity on validation or training data
                eval_loader = val_loader if val_loader is not None else decoder_loader
                perplexity = self.decoder_trainer.compute_perplexity(eval_loader)
                logger.info(f"Decoder Epoch {epoch}: Loss = {avg_loss:.6f}, Perplexity = {perplexity:.2f}")
                logger.info(f"Decoder Epoch {epoch}: Loss = {avg_loss:.6f}")

        logger.info("GPT-2 decoder training completed!")


    def generate_sequence(
            self,
            reference_embedding: torch.Tensor,
            antigen_embeddings: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            num_steps: int = 100,
            noise_scale: float = 1.0,
            max_len: int = None,
    ) -> str:
        """
        Generate a single antibody sequence from a reference embedding using Heun sampling.

        Args:
            reference_embedding: 1D or 2D tensor [hidden_dim] or [1, hidden_dim]
            antigen_embeddings:  1D or 2D tensor [hidden_dim] or [1, hidden_dim]
            temperature:         Sampling temperature (lower = more conservative)
            top_k:               Top-k tokens to sample from (0 = disabled)
            top_p:               Nucleus sampling threshold
            num_steps:           Number of ODE integration steps
            noise_scale:         Scale of initial noise in flow model
            max_len:             Max sequence length (defaults to self.max_seq_len)

        Returns:
            Decoded amino acid sequence string
        """
        max_len = max_len or self.max_seq_len
        vocab_size = self.decoder.vocab_size
        effective_top_k = min(top_k, vocab_size - 1) if top_k > 0 else 0

        # Ensure [1, hidden_dim]
        if reference_embedding.dim() == 1:
            reference_embedding = reference_embedding.unsqueeze(0)
        if antigen_embeddings.dim() == 1:
            antigen_embeddings = antigen_embeddings.unsqueeze(0)

        with torch.no_grad():
            latent = self.flow_model.heun_sample(
                reference_embedding,
                antigen_emb=antigen_embeddings,
                num_steps=num_steps,
                noise_scale=noise_scale,
            )

            generated_tokens = self.decoder.generate(
                latent,
                max_length=max_len,
                temperature=temperature,
                top_k=effective_top_k,
                top_p=top_p,
            )

        sequence = self.tokens_to_sequence(generated_tokens.squeeze(0))
        return sequence

    def generate_multiple_sequences(
            self,
            reference_embedding: torch.Tensor,
            antigen_embeddings: torch.Tensor,
            num_sequences: int,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            num_steps: int = 100,
            noise_scale: float = 1.0,
            guidance_scale: float = 0.0,
            max_len: int = None,
            batch_size: int = 10,
            show_progress: bool = True,
    ) -> list[str]:
        """
        Generate multiple antibody sequences from a reference embedding.

        Args:
            reference_embedding: [hidden_dim] or [1, hidden_dim]
            antigen_embeddings:  [hidden_dim] or [1, hidden_dim]
            num_sequences:       Total number of sequences to generate
            temperature:         Sampling temperature
            top_k:               Top-k filtering (0 = disabled)
            top_p:               Nucleus sampling threshold
            num_steps:           ODE integration steps for Heun sampler
            noise_scale:         Initial noise scale
            max_len:             Max sequence length (defaults to self.max_seq_len)
            batch_size:          How many sequences to generate per batch
            show_progress:       Whether to display a tqdm progress bar

        Returns:
            List of generated sequence strings, length == num_sequences
        """
        max_len = max_len or self.max_seq_len
        vocab_size = self.decoder.vocab_size
        effective_top_k = min(top_k, vocab_size - 1) if top_k > 0 else 0

        # Normalize embeddings once — reused across all batches
        if reference_embedding.dim() == 1:
            reference_embedding = reference_embedding.unsqueeze(0)
        if antigen_embeddings.dim() == 1:
            antigen_embeddings = antigen_embeddings.unsqueeze(0)

        sequences = []
        batches = range(0, num_sequences, batch_size)

        if show_progress:
            from tqdm import tqdm
            batches = tqdm(batches, desc="Generating sequences", unit="batch")

        with torch.no_grad():
            for batch_start in batches:
                current_batch_size = min(batch_size, num_sequences - batch_start)

                # Expand embeddings to [batch_size, hidden_dim] for batched sampling
                ref_batch = reference_embedding.expand(current_batch_size, -1)
                ant_batch = antigen_embeddings.expand(current_batch_size, -1)

                # Sample latents for the whole batch at once
                latents = self.flow_model.heun_sample(
                    ref_batch,
                    antigen_emb=ant_batch,
                    num_steps=num_steps,
                    noise_scale=noise_scale,
                    guidance_scale=guidance_scale,
                )  # [batch_size, hidden_dim]

                # Decode each latent in the batch
                for i in range(current_batch_size):
                    generated_tokens = self.decoder.generate(
                        latents[i].unsqueeze(0),
                        max_length=max_len,
                        temperature=temperature,
                        top_k=effective_top_k,
                        top_p=top_p,
                    )
                    sequences.append(self.tokens_to_sequence(generated_tokens.squeeze(0)))

        return sequences
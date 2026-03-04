"""
ReconPruner: Visual Token Pruner based on MAE-style reconstruction.

Adapted from FastDriveVLA (arxiv 2507.23318) for SimLingo.

Architecture:
  - PrunerLayer: a single Transformer decoder layer that fuses a learnable
    query token with the visual token sequence to produce saliency context.
  - Scorer: a single linear layer that maps the hadamard product of each
    visual token with the query output to a scalar importance score.
  - Top-K selection: at inference keeps the top K=floor(N*(1-p)) tokens.

Training loss (adversarial MAE-style, no segmentation masks required):
  - ReconDecoder takes the K *selected* tokens + learnable mask tokens for
    the N-K pruned positions and reconstructs all N original visual tokens.
  - A complementary decoder uses the *discarded* N-K tokens to reconstruct
    all N tokens; its loss is *subtracted* (adversarial term) so the scorer
    learns to discard tokens that carry little reconstruction signal.
  - Total: L_fg + alpha * (margin - L_bg)  clipped at 0 so adversarial term
    only activates once the background decoder is sufficiently worse.
  - L_fg = MSE(reconstructed, target) on selected tokens
  - L_bg = MSE(reconstructed_bg, target) on discarded tokens
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Straight-Through top-k mask (enables gradient flow during training)
# ---------------------------------------------------------------------------

class _TopKSTE(torch.autograd.Function):
    """Binary top-k mask with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Return a binary mask [B, N] with the top-k positions set to 1."""
        B, N = scores.shape
        _, indices = torch.topk(scores, k, dim=1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, indices, 1.0)
        return mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass gradient through unchanged
        return grad_output, None


def topk_mask_ste(scores: torch.Tensor, k: int) -> torch.Tensor:
    return _TopKSTE.apply(scores, k)


# ---------------------------------------------------------------------------
# ReconDecoder: reconstruct all N visual tokens from a (possibly incomplete)
# subset of tokens, using learnable mask tokens for missing positions.
# ---------------------------------------------------------------------------

class ReconDecoder(nn.Module):
    """
    Lightweight MAE-style decoder.

    Given K attended tokens (with their original positions) and N-K mask
    tokens, reconstruct all N original visual tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        tokens: torch.Tensor,          # [B, K, D]  selected/discarded tokens
        token_indices: torch.Tensor,   # [B, K]     their original positions in [0, N)
        total_tokens: int,             # N
    ) -> torch.Tensor:
        """
        Returns reconstructed sequence of shape [B, N, D].
        Positions not covered by `token_indices` are filled with mask_token.
        """
        B, K, D = tokens.shape

        # Build full sequence with mask tokens at missing positions
        full_seq = self.mask_token.expand(B, total_tokens, D).clone()
        # Scatter the available tokens back to their original positions
        idx_expanded = token_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        full_seq.scatter_(1, idx_expanded, tokens)

        out = self.decoder(full_seq)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# ReconPruner
# ---------------------------------------------------------------------------

class ReconPruner(nn.Module):
    """
    Plug-and-play visual token pruner for VLA models.

    Args:
        hidden_size:         Dimension of visual tokens (after LLM projection).
        pruning_ratio:       Fraction of tokens to discard (0 = keep all, 0.5 = keep half).
        num_pruner_heads:    Multi-head attention heads in the PrunerLayer.
        num_decoder_layers:  Depth of the reconstruction decoder (used during training).
        adversarial_margin:  Margin hyperparameter for adversarial term.
        adversarial_alpha:   Weight of adversarial term in training loss.
    """

    def __init__(
        self,
        hidden_size: int,
        pruning_ratio: float = 0.5,
        num_pruner_heads: int = 8,
        num_decoder_layers: int = 2,
        adversarial_margin: float = 0.1,
        adversarial_alpha: float = 0.5,
    ):
        super().__init__()
        assert 0.0 <= pruning_ratio < 1.0, "pruning_ratio must be in [0, 1)"

        self.hidden_size = hidden_size
        self.pruning_ratio = pruning_ratio
        self.adversarial_margin = adversarial_margin
        self.adversarial_alpha = adversarial_alpha

        # Learnable query token that captures global saliency context
        self.saliency_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.saliency_query, std=0.02)

        # PrunerLayer: single Transformer decoder layer
        # (query attends over visual tokens as memory)
        self.pruner_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_pruner_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True,
        )

        # Scorer: maps element-wise (token * query_out) → scalar
        self.scorer = nn.Linear(hidden_size, 1)

        # Reconstruction decoders (only used during training)
        self.fg_decoder = ReconDecoder(hidden_size, num_decoder_layers, num_pruner_heads)
        self.bg_decoder = ReconDecoder(hidden_size, num_decoder_layers, num_pruner_heads)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """Compute importance score per token. Returns [B, N]."""
        B = visual_tokens.size(0)
        q = self.saliency_query.expand(B, -1, -1)       # [B, 1, D]
        # Query attends over visual tokens as memory
        q_out = self.pruner_layer(q, visual_tokens)    # [B, 1, D]
        # Broadcast query output across all tokens, then element-wise product
        saliency = visual_tokens * q_out               # [B, N, D]
        scores = self.scorer(saliency).squeeze(-1)     # [B, N]
        return scores

    # ------------------------------------------------------------------
    # Forward (inference) – prune and return selected tokens
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(
        self,
        visual_tokens: torch.Tensor,   # [B, N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pruned_tokens : [B, K, D]  top-K tokens sorted by importance
            keep_indices  : [B, K]     original positions
        """
        B, N, D = visual_tokens.shape
        K = max(1, int(N * (1.0 - self.pruning_ratio)))

        scores = self.score_tokens(visual_tokens)          # [B, N]
        _, keep_indices = torch.topk(scores, K, dim=1)    # [B, K]
        keep_indices, _ = keep_indices.sort(dim=1)         # preserve spatial order

        idx_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, D)
        pruned_tokens = visual_tokens.gather(1, idx_expanded)  # [B, K, D]
        return pruned_tokens, keep_indices

    # ------------------------------------------------------------------
    # Forward (training) – prune + compute reconstruction loss
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_tokens: torch.Tensor,          # [B, N, D]
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            visual_tokens: post-projection visual embeddings [B, N, D]
            training:      if True, also compute reconstruction loss

        Returns:
            pruned_tokens : [B, K, D]
            recon_loss    : scalar tensor or None
        """
        B, N, D = visual_tokens.shape
        K = max(1, int(N * (1.0 - self.pruning_ratio)))

        scores = self.score_tokens(visual_tokens)          # [B, N]

        if not training:
            _, keep_indices = torch.topk(scores, K, dim=1)
            keep_indices, _ = keep_indices.sort(dim=1)
            idx_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, D)
            pruned_tokens = visual_tokens.gather(1, idx_expanded)
            return pruned_tokens, None

        # ---- Training path ----
        # Differentiable top-k mask via Straight-Through Estimator
        fg_mask = topk_mask_ste(scores, K)                 # [B, N]  1 = keep
        bg_mask = 1.0 - fg_mask                            # [B, N]  1 = discard

        # Gather foreground and background tokens
        fg_idx = fg_mask.bool()
        # We need consistent tensor shapes: use sort + gather
        _, sorted_all = torch.sort(scores, dim=1, descending=True)
        fg_indices = sorted_all[:, :K]                     # [B, K]
        bg_indices = sorted_all[:, K:]                     # [B, N-K]
        fg_indices, _ = fg_indices.sort(dim=1)
        bg_indices, _ = bg_indices.sort(dim=1)

        idx_fg = fg_indices.unsqueeze(-1).expand(-1, -1, D)
        idx_bg = bg_indices.unsqueeze(-1).expand(-1, -1, D)
        fg_tokens = visual_tokens.gather(1, idx_fg)        # [B, K, D]
        bg_tokens = visual_tokens.gather(1, idx_bg)        # [B, N-K, D]

        # Reconstruct all N tokens from foreground tokens (should be easy)
        recon_fg = self.fg_decoder(fg_tokens, fg_indices, N)   # [B, N, D]
        # Reconstruct all N tokens from background tokens (should be harder)
        recon_bg = self.bg_decoder(bg_tokens, bg_indices, N)   # [B, N, D]

        # Target: detached original visual tokens
        target = visual_tokens.detach()

        # MSE reconstruction loss for foreground (minimize)
        loss_fg = F.mse_loss(recon_fg, target)
        # MSE reconstruction loss for background (maximise via adversarial margin)
        loss_bg = F.mse_loss(recon_bg, target)
        # Adversarial term: penalise if background decoder is too good
        loss_adv = torch.clamp(self.adversarial_margin - (loss_bg - loss_fg), min=0.0)

        recon_loss = loss_fg + self.adversarial_alpha * loss_adv

        # Pruned output (use hard top-k, gradients flow through scorer via STE)
        pruned_tokens = fg_tokens
        return pruned_tokens, recon_loss

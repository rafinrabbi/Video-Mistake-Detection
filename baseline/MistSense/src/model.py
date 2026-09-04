"""MistSense — model architecture.

Paper: Patsch et al., "MistSense: Versatile Online Detection of Procedural
and Execution Mistakes" (ICCV 2025).
URL: https://openaccess.thecvf.com/content/ICCV2025/papers/Patsch_MistSense_Versatile_Online_Detection_of_Procedural_and_Execution_Mistakes_ICCV_2025_paper.pdf

Section references throughout:
  §3.1 — Mistake Detection (dual-stream Q-Former architecture)
  §3.2 — Error Explanation (gated routing to LLM)
  §3.3 — Learning Objectives (three training stages)
  §4.6 — Implementation Settings (ViT-L, Llama2-7B, hyperparameters)

This file implements the architecture only. Loss is in `loss.py`,
training in `train.py`. Pretrained backbones (ViT-L, BLIP-2 Q-Former,
Llama-2-7B) are loaded from external libraries — we do NOT reimplement them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """All architecture knobs. Sourced from configs/base.yaml.

    Citations to paper §:
        d_visual, d_model, num_queries, segment_length — §3.1
        num_joints, num_hands, coord_dim, d_hand — §3.1
        llm_name, llm_hidden — §4.6
        detection_threshold — §3.2 (τ)
        lora_* — §3.3
    """

    # Visual branch (§3.1)
    visual_encoder: str = "eva_vit_l"
    base_qformer_pretrained: str = "Salesforce/blip2-opt-2.7b"
    d_visual: int = 1024            # `d_1` — ViT-L patch dim [UNSPECIFIED value]
    d_model: int = 768              # `d_2` — BLIP-2 default [UNSPECIFIED value]
    num_queries: int = 32           # `t_q` — BLIP-2 default [UNSPECIFIED value]
    segment_length: int = 8         # `t_s` — [UNSPECIFIED value]
    video_qformer_layers: int = 12  # BERT-base default [UNSPECIFIED]
    video_qformer_heads: int = 12   # BERT-base default [UNSPECIFIED]

    # Hand pose branch (§3.1)
    num_joints: int = 21            # `v` — Mediapipe [23]
    num_hands: int = 2              # `m`
    coord_dim: int = 3              # `c` — (x,y,z) [UNSPECIFIED]
    d_hand: int = 256               # `d_h` [UNSPECIFIED]
    tcn_channels: int = 128         # [UNSPECIFIED]
    tcn_layers: int = 3             # [UNSPECIFIED]

    # Head + LLM
    num_classes: int = 1            # §3.1 — `m ∈ R^{t_q}` binary
    llm_name: str = "meta-llama/Llama-2-7b-hf"
    llm_hidden: int = 4096          # Llama2-7B embedding dim
    detection_threshold: float = 0.5  # §3.2 — `τ` [UNSPECIFIED value]

    # LoRA (§3.3)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")


# ---------------------------------------------------------------------------
# §3.1 — Q-Former (Video + Hand share the same structure)
# ---------------------------------------------------------------------------
class QFormerBlock(nn.Module):
    """One transformer block used inside the Video / Hand Q-Former.

    §3.1: "the Video Q-Former ... consists of a BERT encoder [6] ...
    retains the overall structure of [20]". We implement a minimal BERT-style
    block with self-attention over queries + cross-attention to features.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross-attention from queries to features (kv comes from `feats`)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),                              # [UNSPECIFIED] activation — GELU per BERT default
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        q:     (B, t_q, d_model) — learnable queries
        feats: (B, T, d_model)   — encoded features (visual or hand)
        out:   (B, t_q, d_model)
        """
        # Self-attn on queries
        h, _ = self.self_attn(q, q, q)                                 # (B, t_q, d_model)
        q = self.norm1(q + self.dropout(h))
        # Cross-attn: queries attend to features
        h, _ = self.cross_attn(q, feats, feats)                        # (B, t_q, d_model)
        q = self.norm2(q + self.dropout(h))
        # FFN
        q = self.norm3(q + self.dropout(self.ffn(q)))                   # (B, t_q, d_model)
        return q


class QFormer(nn.Module):
    """§3.1 — Video Q-Former (and Hand Q-Former, identical structure).

    "In combination with learnable queries q_rgb ∈ R^{t_q × d_2}, the
    features v are input to the Video Q-Former. The learnable queries are
    trainable token embeddings that guide the model to extract task-relevant
    information from features with cross-attention."
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(cfg.num_queries, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [QFormerBlock(cfg.d_model, cfg.video_qformer_heads) for _ in range(cfg.video_qformer_layers)]
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, T, d_model) — projected visual or hand features over time
        out:   (B, t_q, d_model) — `f_rgb` or `f_hand`
        """
        B = feats.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)                # (B, t_q, d_model)
        for blk in self.blocks:
            q = blk(q, feats)
        return q


# ---------------------------------------------------------------------------
# §3.1 — Hand pose encoder (TCN per joint + self-attention combine)
# ---------------------------------------------------------------------------
class JointTCN(nn.Module):
    """Per-joint temporal conv stack.

    §3.1: "Temporal Convolutional Networks (TCNs) [17] that model the
    separate joints as well as the wrist individually". Following [32]
    (Shamil 2024, "On the utility of 3D hand poses for action recognition").

    [PARTIALLY_SPECIFIED] Exact TCN depth/width not stated; defaults from
    config.
    """

    def __init__(self, in_ch: int, hid_ch: int, n_layers: int):
        super().__init__()
        layers = []
        c_in = in_ch
        for i in range(n_layers):
            layers += [
                nn.Conv1d(c_in, hid_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(hid_ch),
                nn.ReLU(inplace=True),                                  # [UNSPECIFIED] activation
            ]
            c_in = hid_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, c, T) -> (B, hid_ch, T)"""
        return self.net(x)


class HandPoseEncoder(nn.Module):
    """§3.1 — Hand pose encoder, adapted from [32].

    Input:  h ∈ R^{c × t_s × v × m}  (coord, time, joints, hands)
    Output: h1 ∈ R^{t_s × d_h}        (per-frame hand feature)

    Procedure (per paper):
      1. Per-joint TCNs (incl. dedicated wrist TCN)
      2. Self-attention combines joint embeddings into a single per-frame token
      3. Linear projects to d_h
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_joints = cfg.num_joints
        self.num_hands = cfg.num_hands
        # one TCN per (hand, joint) — share weights across hands, separate per joint
        self.joint_tcns = nn.ModuleList(
            [JointTCN(cfg.coord_dim * cfg.num_hands, cfg.tcn_channels, cfg.tcn_layers) for _ in range(cfg.num_joints)]
        )
        # Self-attention over joints (per-frame)
        self.joint_attn = nn.MultiheadAttention(cfg.tcn_channels, num_heads=4, batch_first=True)
        self.proj = nn.Linear(cfg.tcn_channels, cfg.d_hand)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, c, T, V, M)  — c=coord_dim, T=segment_length, V=joints, M=hands
        out: (B, T, d_hand) — `h_1`
        """
        B, C, T, V, M = h.shape
        # Per-joint TCN inputs: combine hands as channels (c*M, T) per joint v
        joint_feats = []
        for v in range(V):
            x = h[:, :, :, v, :]                       # (B, c, T, M)
            x = x.permute(0, 1, 3, 2).reshape(B, C * M, T)  # (B, c*M, T)
            x = self.joint_tcns[v](x)                   # (B, hid, T)
            joint_feats.append(x)
        # Stack: (B, V, hid, T) -> (B, T, V, hid)
        joint_feats = torch.stack(joint_feats, dim=1).permute(0, 3, 1, 2)
        B_, T_, V_, H_ = joint_feats.shape
        # Self-attention over joints, per frame
        flat = joint_feats.reshape(B_ * T_, V_, H_)     # (B*T, V, hid)
        attended, _ = self.joint_attn(flat, flat, flat)  # (B*T, V, hid)
        # Mean over joints to get per-frame feature
        framewise = attended.mean(dim=1)                # (B*T, hid)
        framewise = framewise.reshape(B_, T_, H_)       # (B, T, hid)
        return self.proj(framewise)                     # (B, T, d_hand)


# ---------------------------------------------------------------------------
# §3.1 — MistSense main module
# ---------------------------------------------------------------------------
class MistSense(nn.Module):
    """§3.1 + §3.2 — MistSense full forward.

    NOTE: ViT-L and the base BLIP-2 Q-Former are loaded from external
    pretrained checkpoints in `build_mistsense()`. Here we assume the
    visual features `v ∈ R^{t_s × d_1}` are already extracted (caller can
    skip the frozen backbones at training time by precomputing features).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # §3.1 — visual feature projection to d_model. Paper does not state
        # explicitly but `v ∈ R^{t_s×d_1}` differs from `d_2`, so a projection
        # is required before the Video Q-Former.
        self.visual_proj = nn.Linear(cfg.d_visual, cfg.d_model)
        # §3.1 — framewise positional embedding (added before Q-Former)
        self.video_pos_emb = nn.Parameter(torch.zeros(1, cfg.segment_length, cfg.d_model))
        # §3.1 — "Video Q-Former"
        self.video_qformer = QFormer(cfg)

        # §3.1 — hand pose encoder + projection + PE + Hand Q-Former
        self.hand_encoder = HandPoseEncoder(cfg)
        self.hand_proj = nn.Linear(cfg.d_hand, cfg.d_model)
        self.hand_pos_emb = nn.Parameter(torch.zeros(1, cfg.segment_length, cfg.d_model))
        self.hand_qformer = QFormer(cfg)

        # §3.1 — mistake classification layer
        # "The features f_rgb and f_hand are then added together and passed
        # to the mistake classification layer to obtain the final mistake
        # logits m ∈ R^{t_q}."
        self.mistake_head = nn.Linear(cfg.d_model, cfg.num_classes)

        # §3.2 — projection from Video Q-Former features to LLM embedding space
        self.llm_proj = nn.Linear(cfg.d_model, cfg.llm_hidden)

    def encode_video(self, v_feat: torch.Tensor) -> torch.Tensor:
        """
        v_feat: (B, t_s, d_1) — pre-extracted ViT-L+base-Q-Former features
        returns f_rgb: (B, t_q, d_model)
        """
        x = self.visual_proj(v_feat) + self.video_pos_emb              # (B, t_s, d_model)
        return self.video_qformer(x)

    def encode_hand(self, h_input: torch.Tensor) -> torch.Tensor:
        """
        h_input: (B, c, t_s, v, m)
        returns f_hand: (B, t_q, d_model)
        """
        h1 = self.hand_encoder(h_input)                                # (B, t_s, d_hand)
        h2 = self.hand_proj(h1) + self.hand_pos_emb                    # (B, t_s, d_model)
        return self.hand_qformer(h2)

    def forward(
        self,
        v_feat: torch.Tensor,
        h_input: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """§3.1 forward.

        v_feat:  (B, t_s, d_1)
        h_input: (B, c, t_s, v, m) or None for RGB-only ablation

        Returns dict:
          'logits':  (B, t_q)            — per-query mistake logits `m`
          'f_rgb':   (B, t_q, d_model)   — Video Q-Former output (for gating to LLM)
          'gate':    (B,)                — sigmoid-aggregated mistake confidence g_m
        """
        f_rgb = self.encode_video(v_feat)                              # (B, t_q, d_model)
        if h_input is not None:
            f_hand = self.encode_hand(h_input)                         # (B, t_q, d_model)
            f = f_rgb + f_hand                                         # §3.1 — summed fusion
        else:
            f = f_rgb
        logits = self.mistake_head(f).squeeze(-1)                      # (B, t_q)
        # §3.2 — gating. Paper: "where σ is the sigmoid function".
        # Aggregate over queries by max — paper equation omitted in PDF;
        # using max is the natural per-segment "is there any mistake?"
        # signal. [PARTIALLY_SPECIFIED] aggregation method.
        gate = torch.sigmoid(logits).max(dim=-1).values                # (B,)
        return {"logits": logits, "f_rgb": f_rgb, "gate": gate}

    def project_to_llm(self, f_rgb: torch.Tensor) -> torch.Tensor:
        """§3.2 — route Video Q-Former features into Llama2 embedding space.

        f_rgb: (B, t_q, d_model)
        out:   (B, t_q, llm_hidden)  — visual prefix tokens for the LLM
        """
        return self.llm_proj(f_rgb)

    def should_explain(self, gate: torch.Tensor) -> torch.Tensor:
        """§3.2 — boolean mask: gate ≥ τ -> route to LLM."""
        return gate >= self.cfg.detection_threshold


# ---------------------------------------------------------------------------
# Factory / wiring with pretrained backbones (optional dependency)
# ---------------------------------------------------------------------------
def build_mistsense(cfg: ModelConfig) -> MistSense:
    """Construct MistSense. Pretrained ViT-L / base Q-Former / Llama2 are
    loaded by the trainer (see `train.py`) — this factory only builds the
    trainable subnet. Keeping pretrained loading external keeps `model.py`
    importable without huggingface_hub network access for tests.
    """
    return MistSense(cfg)

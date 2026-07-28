"""EEG-to-MusicCoCa alignment model.

Pretrained LaBraM (official braindecode checkpoint) as EEG encoder, with an
MNE-interpolated channel projection for arbitrary montages
(InterpolatedLaBraM), followed by a trainable alignment head into the
768-dim MusicCoCa style space. Trained with a CLIP-style symmetric
contrastive loss (ported from the EEG-to-Music reference implementation).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MUSICCOCA_DIM = 768


class EEGMusicAligner(nn.Module):
    """LaBraM backbone + projection head into MusicCoCa space."""

    def __init__(self, chs_info, out_dim=MUSICCOCA_DIM, freeze_backbone=True,
                 unfreeze_last_blocks=0, n_times=2000):
        super().__init__()
        from braindecode.models import Labram, InterpolatedLaBraM

        base = Labram.from_pretrained("braindecode/labram-pretrained")
        base.reset_classifier(0)

        backbone = InterpolatedLaBraM(chs_info=chs_info, n_times=n_times, n_outputs=1)
        sd = base.state_dict()
        ckpt_temporal = sd.pop("temporal_embedding")
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        ignored = ("interpolation_layer", "final_layer", "temporal_embedding")
        missing = [k for k in missing if not k.startswith(ignored)]
        if missing or unexpected:
            raise RuntimeError(
                f"LaBraM weight mismatch. Missing: {missing}, unexpected: {unexpected}")
        with torch.no_grad():
            n_rows = backbone.temporal_embedding.shape[1]
            backbone.temporal_embedding.copy_(ckpt_temporal[:, :n_rows])

        self.backbone = backbone
        embed_dim = base.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last_blocks > 0:
                for blk in self.backbone.blocks[-unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True
                if self.backbone.fc_norm is not None:
                    for p in self.backbone.fc_norm.parameters():
                        p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def trainable_parameters(self, backbone_lr_mult=0.1):
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        bb_params = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr_mult": 1.0}]
        if bb_params:
            groups.append({"params": bb_params, "lr_mult": backbone_lr_mult})
        return groups

    def forward(self, x):
        out = self.backbone(x, return_features=True)
        return self.head(out["cls_token"])


class CLIPLoss(nn.Module):
    """Symmetric contrastive loss with learnable temperature."""

    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1.0 / init_temperature), dtype=torch.float32))

    def forward(self, z_eeg, z_audio):
        z_eeg = F.normalize(z_eeg, dim=-1)
        z_audio = F.normalize(z_audio, dim=-1)
        logits = self.logit_scale.exp() * z_eeg @ z_audio.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def retrieval_recall(z_eeg, z_audio, ks=(1, 5)):
    """Recall@K of EEG->audio retrieval over the batch/eval set."""
    z_eeg = F.normalize(z_eeg, dim=-1)
    z_audio = F.normalize(z_audio, dim=-1)
    sim = z_eeg @ z_audio.t()
    n = sim.shape[0]
    ranks = sim.argsort(dim=1, descending=True)
    targets = torch.arange(n, device=sim.device).unsqueeze(1)
    hits = (ranks == targets).nonzero()[:, 1]
    out = {}
    for k in ks:
        k = min(k, n)
        out[f"R@{k}"] = (hits < k).float().mean().item()
    return out

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vit as custom_vit


class PrefixViTBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        prefix_len: int = 5,
        num_prefix_layers: int = 5,
        use_norga: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        assert hasattr(custom_vit, backbone_name), f"Unsupported backbone {backbone_name} for prefix ViT"
        self.backbone = getattr(custom_vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        self.task_num = task_num
        self.prefix_len = prefix_len
        self.num_prefix_layers = num_prefix_layers
        self.use_norga = use_norga

        self.embed_dim = self.backbone.num_features
        self.num_heads = self.backbone.blocks[0].attn.num_heads
        head_dim = self.embed_dim // self.num_heads
        k_shape = (num_prefix_layers, task_num, prefix_len, self.num_heads, head_dim)
        self.prefix_k = nn.Parameter(torch.zeros(k_shape))
        self.prefix_v = nn.Parameter(torch.zeros(k_shape))
        nn.init.trunc_normal_(self.prefix_k, std=0.02)
        nn.init.trunc_normal_(self.prefix_v, std=0.02)

        if self.use_norga:
            # act_scale: [num_layers, 2, 1, 1]
            self.act_scale = nn.Parameter(torch.ones(num_prefix_layers, 2, 1, 1))
            self.gate_act = nn.Sigmoid()
        else:
            self.act_scale = None
            self.gate_act = None

    @torch.no_grad()
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def freeze_act_scale(self) -> None:
        if self.act_scale is not None:
            self.act_scale.requires_grad_(False)

    def _apply_prefix_attn(self, x_norm, block, layer_idx: int, task_ids: torch.Tensor) -> torch.Tensor:
        B, N, C = x_norm.shape
        attn = block.attn
        qkv = attn.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # prefix for current layer & task (per-sample)
        task_ids = task_ids.to(x_norm.device).long().clamp(0, self.task_num - 1)
        # prefix_k[layer_idx]: [T, L, H, D]
        k_layer = self.prefix_k[layer_idx]
        v_layer = self.prefix_v[layer_idx]
        k_pref = k_layer[task_ids]  # [B, L, H, D]
        v_pref = v_layer[task_ids]
        k_pref = k_pref.permute(0, 2, 1, 3)  # [B, H, L, D]
        v_pref = v_pref.permute(0, 2, 1, 3)
        k_cat = torch.cat([k_pref, k], dim=2)
        v_cat = torch.cat([v_pref, v], dim=2)
        scale = attn.scale
        attn_logits = (q @ k_cat.transpose(-2, -1)) * scale
        if self.use_norga and self.act_scale is not None:
            s = self.act_scale[layer_idx]
            # apply NoRGa gating only on prefix (first prefix_len keys) along last dim
            prompt_part = attn_logits[..., : self.prefix_len]
            base_part = attn_logits[..., self.prefix_len :]
            prompt_part = prompt_part + self.gate_act(prompt_part * s[0]) * s[1]
            attn_logits = torch.cat([prompt_part, base_part], dim=-1)
        attn_prob = attn_logits.softmax(dim=-1)
        attn_prob = attn.attn_drop(attn_prob)
        out = (attn_prob @ v_cat).transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out

    def forward_features(self, x: torch.Tensor, use_prefix: bool, task_id: Optional[int] = None) -> torch.Tensor:
        b = self.backbone
        x = b.patch_embed(x)
        B, N, _ = x.shape
        cls_tokens = b.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + b.pos_embed
        x = b.pos_drop(x)
        task_ids_tensor = None
        if use_prefix and task_id is not None:
            if isinstance(task_id, int):
                task_ids_tensor = torch.full((B,), task_id, device=x.device, dtype=torch.long)
            elif isinstance(task_id, torch.Tensor):
                if task_id.dim() == 0:
                    task_ids_tensor = task_id.view(1).expand(B).to(x.device).long()
                else:
                    assert task_id.shape[0] == B, "task_id tensor must have shape [B]"
                    task_ids_tensor = task_id.to(x.device).long()
            else:
                raise TypeError("task_id must be int or 1D/0D tensor when use_prefix is True")
        for i, blk in enumerate(b.blocks):
            if use_prefix and i < self.num_prefix_layers and task_ids_tensor is not None:
                x_norm = blk.norm1(x)
                attn_out = self._apply_prefix_attn(x_norm, blk, i, task_ids_tensor)
                x = x + blk.drop_path1(blk.ls1(attn_out))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            else:
                x = blk(x)
        x = b.norm(x)
        return x[:, 0]


class HiDePrefixModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        prefix_len: int = 5,
        num_prefix_layers: int = 5,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = PrefixViTBackbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            prefix_len=prefix_len,
            num_prefix_layers=num_prefix_layers,
            use_norga=False,
            pretrained=pretrained,
        )
        # freeze all pre-trained ViT backbone parameters; only prefixes and heads are trainable
        if hasattr(self.backbone, "backbone"):
            for name, param in self.backbone.backbone.named_parameters():
                param.requires_grad = False
        D = self.backbone.embed_dim
        self.feature_dim = D
        self.prompt_head = nn.Linear(D, num_classes)
        self.g_mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU(inplace=True))
        self.g_head = nn.Linear(D, num_classes)

        # statistics for prompt branch (backbone features)
        self.register_buffer("p_count", torch.zeros(num_classes))
        self.register_buffer("p_sum", torch.zeros(num_classes, D))
        self.register_buffer("p_sum_xxT", torch.zeros(num_classes, D, D))
        # statistics for gating branch (g_mlp features)
        self.register_buffer("g_count", torch.zeros(num_classes))
        self.register_buffer("g_sum", torch.zeros(num_classes, D))
        self.register_buffer("g_sum_xxT", torch.zeros(num_classes, D, D))

    def _get_stats(self, branch: str):
        if branch == "prompt":
            return self.p_count, self.p_sum, self.p_sum_xxT
        elif branch == "gate":
            return self.g_count, self.g_sum, self.g_sum_xxT
        else:
            raise ValueError(f"Unknown branch {branch}")

    @torch.no_grad()
    def _update_stats(self, feat: torch.Tensor, labels: torch.Tensor, branch: str) -> None:
        counts, sums, sums_xxT = self._get_stats(branch)
        labels = labels.long()
        unique_labels = labels.unique()
        for c in unique_labels.tolist():
            mask = labels == c
            f_c = feat[mask]
            if f_c.numel() == 0:
                continue
            n_c = f_c.size(0)
            sums[c] += f_c.sum(dim=0)
            counts[c] += n_c
            sums_xxT[c] += f_c.t() @ f_c

    def forward_prompt(
        self,
        x: torch.Tensor,
        task_id: int,
        labels: Optional[torch.Tensor] = None,
        update_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x, use_prefix=True, task_id=task_id)
        logit = self.prompt_head(feat)
        if update_stats and labels is not None:
            self._update_stats(feat.detach(), labels, branch="prompt")
        return logit, feat

    def forward_gate(
        self,
        x: torch.Tensor,
        detach_backbone: bool = True,
        labels: Optional[torch.Tensor] = None,
        update_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x, use_prefix=False, task_id=None)
        if detach_backbone:
            feat = feat.detach()
        h = self.g_mlp(feat)
        logit = self.g_head(h)
        if update_stats and labels is not None:
            self._update_stats(h.detach(), labels, branch="gate")
        return logit, h

    def compute_orth_loss(
        self,
        feat: torch.Tensor,
        old_class_mask: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        counts, sums, _ = self._get_stats(branch)
        device = feat.device
        counts = counts.to(device)
        sums = sums.to(device)
        old_class_mask = old_class_mask.to(device)
        valid = (counts > 0) & old_class_mask
        idx = valid.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return torch.tensor(0.0, device=device)
        proto = sums[idx] / counts[idx].unsqueeze(-1)
        proto = F.normalize(proto, dim=-1)
        proj = feat @ proto.t()
        loss = (proj ** 2).mean()
        return loss

    def sample_features_for_ca(
        self,
        branch: str,
        num_per_class: int = 10,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        counts, sums, sums_xxT = self._get_stats(branch)
        if device is None:
            device = counts.device
        counts = counts.to(device)
        sums = sums.to(device)
        sums_xxT = sums_xxT.to(device)
        idx = (counts > 1).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return None, None
        D = self.feature_dim
        feats = []
        labels = []
        for c in idx.tolist():
            n_c = counts[c]
            mu = sums[c] / n_c
            ExxT = sums_xxT[c] / n_c
            # use only diagonal covariance for numerical stability
            var = ExxT.diagonal(dim1=-2, dim2=-1) - mu * mu
            var = torch.clamp(var, min=1e-5)
            std = torch.sqrt(var)
            samples = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                num_per_class, D, device=device
            )
            feats.append(samples)
            labels.append(torch.full((num_per_class,), c, device=device, dtype=torch.long))
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels


class NoRGaPrefixModel(HiDePrefixModel):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        prefix_len: int = 5,
        num_prefix_layers: int = 5,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            prefix_len=prefix_len,
            num_prefix_layers=num_prefix_layers,
            pretrained=pretrained,
            **kwargs,
        )
        # replace backbone with NoRGa-enabled one and keep ViT frozen
        self.backbone = PrefixViTBackbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            prefix_len=prefix_len,
            num_prefix_layers=num_prefix_layers,
            use_norga=True,
            pretrained=pretrained,
        )
        if hasattr(self.backbone, "backbone"):
            for name, param in self.backbone.backbone.named_parameters():
                param.requires_grad = False

    def freeze_act_scale(self) -> None:
        self.backbone.freeze_act_scale()


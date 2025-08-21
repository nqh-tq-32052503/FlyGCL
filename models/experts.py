import logging
from typing import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseExpert(nn.Module):
    """Abstract base class for per-task experts.
    Must implement forward(backbone, inputs, expert_ids) -> raw CLS features (no RP).
    """
    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_new_expert(self, task_id: int):
        """Initialize new expert's parameters."""
        raise NotImplementedError


# ---------------------------- PromptExpert (per-task) ---------------------------
class PromptExpert(BaseExpert):
    """Per-task fixed prompt expert.

    Each expert owns a fixed set of prompt tokens per specified layer position.
    No dynamic selection, no key/query mechanism.
    """
    def __init__(self,
                 num_experts: int,
                 len_prompt: int = 20,
                 embed_dim: int = 768):
        super().__init__()
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim
        self.prompts = nn.Parameter(torch.randn(num_experts, len_prompt, embed_dim, requires_grad=True))
        nn.init.uniform_(self.prompts, -1, 1)

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(token_appended + backbone.pos_embed)

        # Create mask for valid expert_ids (>= 0)
        valid_mask = expert_ids >= 0

        if valid_mask.all():
            # All samples have valid expert_ids, use original logic
            prompts = self.prompts[expert_ids.long()]
            prompts = prompts + backbone.pos_embed[:,0].unsqueeze(0).expand(B, self.len_prompt, D)
            x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)
        elif not valid_mask.any():
            # No samples have valid expert_ids, skip prompt concatenation entirely
            pass
        else:
            # Mixed batch: some valid, some invalid expert_ids
            # Need to process separately to maintain correct sequence lengths
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            invalid_indices = (~valid_mask).nonzero(as_tuple=True)[0]

            # Process valid samples with prompts
            if len(valid_indices) > 0:
                x_valid = x[valid_indices]
                valid_expert_ids = expert_ids[valid_indices]
                prompts = self.prompts[valid_expert_ids.long()]
                prompts = prompts + backbone.pos_embed[:,0].unsqueeze(0).expand(len(valid_indices), self.len_prompt, D)
                x_valid = torch.cat((x_valid[:,0].unsqueeze(1), prompts, x_valid[:,1:]), dim=1)

            # Process invalid samples without prompts
            if len(invalid_indices) > 0:
                x_invalid = x[invalid_indices]

            # Process through transformer blocks separately
            if len(valid_indices) > 0:
                x_valid = backbone.blocks(x_valid)
                x_valid = backbone.norm(x_valid)
                cls_valid = x_valid[:, 0]

            if len(invalid_indices) > 0:
                x_invalid = backbone.blocks(x_invalid)
                x_invalid = backbone.norm(x_invalid)
                cls_invalid = x_invalid[:, 0]

            # Reconstruct output in original order
            output = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            if len(valid_indices) > 0:
                output[valid_indices] = cls_valid
            if len(invalid_indices) > 0:
                output[invalid_indices] = cls_invalid

            return output

        x = backbone.blocks(x)
        x = backbone.norm(x)
        return x[:, 0]

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        if expert_id == 0:
            return
        assert expert_id >= 1
        assert expert_id < self.num_experts

        # mean of previous experts
        prev_experts = self.prompts[:expert_id].clone()
        prev_experts_mean = prev_experts.mean(dim=0)
        self.prompts.data[expert_id] = prev_experts_mean


# ------------------------------ LoRAExpert (per-task) ------------------------------
class LoRAExpert(BaseExpert):
    #TODO
    pass
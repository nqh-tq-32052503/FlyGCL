import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from .flyprompt import Prompt

logger = logging.getLogger()


class SPrompt(nn.Module):
    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 100,
        backbone_name: str = None,
        len_prompt: int = 20,
        pos_prompt: Iterable[int] = (0, 1, 2, 3, 4),
        **kwargs,
    ):
        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes
        self.len_prompt = len_prompt
        self.pos_prompt = pos_prompt

        self.task_count = 0

        # Backbone (same as FlyPrompt)
        assert backbone_name is not None, "backbone_name must be specified"
        if hasattr(vit, backbone_name):
            logger.info(f"Using custom ViT model: {backbone_name}")
            self.add_module(
                "backbone",
                getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes),
            )
        else:
            logger.info(f"Using timm model: {backbone_name}")
            self.add_module(
                "backbone",
                timm.create_model(backbone_name, pretrained=True, num_classes=num_classes),
            )

        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

        # Expert prompts (identical structure to FlyPrompt)
        self.experts = Prompt(
            num_experts=self.task_num,
            len_prompt=self.len_prompt,
            embed_dim=self.embed_dim,
            pos_prompt=self.pos_prompt,
        )

    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs):
        if expert_ids is None:
            expert_ids = torch.full(
                (inputs.size(0),),
                self.task_count,
                device=inputs.device,
                dtype=torch.long,
            )
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    @torch.no_grad()
    def process_task_count(self):
        self.task_count += 1
        self.experts.init_new_expert(self.task_count)


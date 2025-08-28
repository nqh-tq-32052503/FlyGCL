import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()


class Prompt(nn.Module):
    def __init__(self,
                 num_experts: int,
                 len_prompt: int = 20,
                 embed_dim: int = 768):
        super().__init__()
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim
        self.prompts = nn.Parameter(
            torch.empty(num_experts, len_prompt, embed_dim)
        )
        nn.init.uniform_(self.prompts, -1, 1)

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(token_appended + backbone.pos_embed)

        prompts = self.prompts[expert_ids.long()]
        prompts = prompts + backbone.pos_embed[:,0].unsqueeze(0).expand(B, self.len_prompt, D)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

        x = backbone.blocks(x)
        x = backbone.norm(x)
        return x[:, 0]

        # x = x[:, 1:self.len_prompt + 1].clone()
        # x = x.mean(dim=1)
        # return x

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        if expert_id == 0 or expert_id == self.num_experts:
            return

        # mean of previous experts
        prev_experts = self.prompts[:expert_id].clone()
        prev_experts_mean = prev_experts.mean(dim=0)
        self.prompts.data[expert_id] = prev_experts_mean


class FlyPrompt(nn.Module):
    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        self.experts = Prompt(
            num_experts=self.task_num,
            len_prompt=30,
            embed_dim=self.embed_dim,
        )

    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1
        self.experts.init_new_expert(self.task_count)
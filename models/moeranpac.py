import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from models.experts import LoRAAttention, LoRAExpert, PromptExpert
from models.ranpac import Adapter

logger = logging.getLogger()


class MoERanPACClassifier(nn.Module):
    def __init__(self,
                 feature_dim  : int,
                 num_classes  : int,
                 use_RP       : bool,
                 M            : int,
                 **kwargs):

        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_RP = use_RP
        self.M = M

        self.fc = nn.Linear(feature_dim, num_classes, bias=False)

        self.rp_initialized = False
        if self.use_RP and self.M > 0:
            self.register_buffer('W_rand', torch.randn(self.feature_dim, self.M))
            self.register_buffer('Q', torch.zeros(self.M, num_classes))
            self.register_buffer('G', torch.zeros(self.M, self.M))
        else:
            self.register_buffer('W_rand', torch.empty(0))
            self.register_buffer('Q', torch.zeros(feature_dim, num_classes))
            self.register_buffer('G', torch.zeros(feature_dim, feature_dim))

    def target2onehot(self, targets, num_classes):
        device = targets.device
        onehot = torch.zeros(targets.size(0), num_classes, device=device)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        return onehot

    def collect_features_labels(self, features, labels):
        features = features.detach()
        labels = labels.detach()

        if self.use_RP:
            features_h = F.relu(features @ self.W_rand)
        else:
            features_h = features

        Y = self.target2onehot(labels, self.num_classes)

        self.Q = self.Q + features_h.T @ Y
        self.G = self.G + features_h.T @ features_h

    def update_classifier(self):
        ridge = 1e4
        device = self.fc.weight.device
        Wo = torch.linalg.solve(self.G + ridge * torch.eye(self.G.size(dim=0), device=device), self.Q).T
        self.fc.weight.data = Wo.to(device)
        self.rp_initialized = True
        logger.info(f"Classifier weights updated using ridge {ridge}")

    def forward_rp_features(self, x):
        if self.use_RP and self.rp_initialized:
            x = F.relu(x @ self.W_rand)
        return x

    def forward_rp_head(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.forward_rp_features(x)
        x = self.forward_rp_head(x)
        return x


class MoERanPAC(nn.Module):
    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 adapter_dim    : int   = 64,
                 ranpac_M       : int   = 10000,
                 ranpac_use_RP  : bool  = True,
                 backbone_name  : str   = None,
                 use_lora       : bool  = True,
                 lora_rank      : int   = 64,
                 lora_alpha     : float = 1.0,
                 lora_layers    : int   = 5,
                 merge_lora     : bool  = True,
                 expert_type    : str   = 'prompt',
                 len_prompt     : int   = 20,
                 expert_layers  : int   = 5,
                 **kwargs):

        super().__init__()

        self.M              = ranpac_M
        self.kwargs         = kwargs
        self.use_RP         = ranpac_use_RP
        self.task_num       = task_num
        self.num_classes    = num_classes
        self.adapter_dim    = adapter_dim
        self.use_lora       = use_lora
        self.lora_rank      = lora_rank
        self.lora_alpha     = lora_alpha
        self.lora_layers    = lora_layers
        self.merge_lora     = merge_lora
        self.expert_type    = expert_type
        self.len_prompt     = len_prompt
        self.expert_layers  = expert_layers

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        if self.use_lora:
            # Apply LoRA to attention layers
            self.lora_attentions = []
            for name, module in self.backbone.named_modules():
                if isinstance(module, vit.Block):
                    # Replace attention with LoRA attention
                    original_attn = module.attn
                    lora_attn = LoRAAttention(original_attn, self.lora_rank, self.lora_alpha)
                    module.attn = lora_attn
                    self.lora_attentions.append(lora_attn)

                    if len(self.lora_attentions) >= self.lora_layers:
                        break

            logger.info(f"LoRA applied to {len(self.lora_attentions)} attention layers with rank={self.lora_rank}, alpha={self.lora_alpha}")
        else:
            # Insert adapter with mlp to each block
            for name, module in self.backbone.named_modules():
                if isinstance(module, vit.Block):
                    module.adapter = Adapter(
                        down_size=self.adapter_dim,
                        n_embd=module.mlp.fc1.in_features,
                        dropout=0.1,
                    )
                    # Create a closure to capture the current module
                    def create_forward_with_adapter(block):
                        def forward_with_adapter(x):
                            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
                            residual = x
                            adapt_x = block.adapter(x)
                            mlp_x = block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
                            x = adapt_x + mlp_x + residual
                            return x
                        return forward_with_adapter

                    module.forward = create_forward_with_adapter(module)

            logger.info("Adapters initialized in all transformer blocks")

        self.classifier = MoERanPACClassifier(
            feature_dim=self.embed_dim,
            num_classes=num_classes,
            use_RP=ranpac_use_RP,
            M=ranpac_M,
        )

        if self.expert_type == 'prompt':
            self.experts = PromptExpert(
                num_experts=self.task_num-1,
                len_prompt=len_prompt,
                embed_dim=self.embed_dim,
            )
        elif self.expert_type == 'lora':
            self.experts = LoRAExpert(
                num_experts=self.task_num-1,
                embed_dim=self.embed_dim,
                num_lora_layers=self.expert_layers,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )

        self.overall_fc = nn.Linear(self.embed_dim, self.num_classes, bias=False)

        self.final_classifier = MoERanPACClassifier(
            feature_dim=self.embed_dim,
            num_classes=num_classes,
            use_RP=ranpac_use_RP,
            M=ranpac_M,
        )

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.forward_features(inputs)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        # Forward pass
        x = self.backbone.forward_features(x)
        x = x[:, 0] # CLS token
        return x
    
    def forward_with_experts(self, x, expert_ids):
        x = self.forward_with_expert_features(x, expert_ids)
        x = self.overall_fc(x)
        return x
    
    def forward_for_final_eval(self, x, expert_ids):
        x = self.forward_with_expert_features(x, expert_ids)
        x = self.final_classifier(x)
        return x
    
    def forward_with_expert_features(self, x, expert_ids):
        x = self.experts(self.backbone, x, expert_ids)
        return x

    def collect_features_labels(self, x, labels):
        with torch.no_grad():
            features = self.forward_features(x)
            # Ensure labels are on same device as features before collection
            if labels.device != features.device:
                labels = labels.to(features.device)
            self.classifier.collect_features_labels(features, labels)

            expert_ids = torch.full((labels.size(0),), self.task_count-1, device=labels.device, dtype=torch.long)
            expert_features = self.forward_with_expert_features(x, expert_ids)
            self.final_classifier.collect_features_labels(expert_features, labels)

    def update_classifier(self):
        self.classifier.update_classifier()
        self.final_classifier.update_classifier()

    def freeze_backbone_except_adapters(self):
        """Freeze backbone except adapters (for adapter mode)"""
        if not self.use_lora:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            for name, module in self.backbone.named_modules():
                if isinstance(module, Adapter):
                    for param in module.parameters():
                        param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.experts.parameters():
            param.requires_grad = False
        for param in self.overall_fc.parameters():
            param.requires_grad = False
        for param in self.final_classifier.parameters():
            param.requires_grad = False

    def freeze_backbone_except_lora(self):
        """Freeze backbone except LoRA parameters (for LoRA mode)"""
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
            # Enable LoRA parameters
            for lora_attn in self.lora_attentions:
                for param in lora_attn.lora_k.parameters():
                    param.requires_grad = True
                for param in lora_attn.lora_v.parameters():
                    param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.experts.parameters():
            param.requires_grad = False
        for param in self.overall_fc.parameters():
            param.requires_grad = False
        for param in self.final_classifier.parameters():
            param.requires_grad = False

    def merge_lora_weights(self):
        """Merge LoRA weights into the original model weights"""
        if not self.use_lora:
            logger.warning("merge_lora_weights called but use_lora=False")
            return

        if not self.merge_lora:
            logger.info("LoRA merging disabled by merge_lora=False, keeping LoRA parameters separate")
            return

        for lora_attn in self.lora_attentions:
            lora_attn.merge_lora_weights()
        logger.info("All LoRA weights merged into backbone")

    def freeze_backbone_except_experts(self):
        """Freeze all parameters except expert parameters"""
        for name, param in self.named_parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.experts.parameters():
            param.requires_grad = True
        for param in self.overall_fc.parameters():
            param.requires_grad = True
        for param in self.final_classifier.parameters():
            param.requires_grad = False

    def copy_classifier(self):
        """copy classifier weights to overall fc"""
        self.overall_fc.weight.data = self.classifier.fc.weight.data.clone()

    def init_new_expert(self, task_id):
        """Initialize new expert's parameters"""
        expert_id = task_id - 1 # note: no expert for first task (task_id=0)
        self.experts.init_new_expert(expert_id)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1
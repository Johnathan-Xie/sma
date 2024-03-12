import math
import random

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

class RandomlySelectedCrossAttentionMasking:
    """Masks based on cross attention"""
    def __init__(
        self,
        masking_ratio=0.5,
        num_per_query=576,
        exclude_seen_reconstruction=True,
        varying_length=True,
        relative_noise=0.0,
        select_initial_ratio=None,
        mask_self=False,
        head_aggregation="mean",
        **kwargs,
    ):
        self.masking_ratio = masking_ratio
        self.num_per_query = num_per_query
        self.exclude_seen_reconstruction = exclude_seen_reconstruction
        self.varying_length = varying_length
        
        self.noise_schedule_scale = 0.0
        self.select_initial_ratio = select_initial_ratio
        self.mask_self = mask_self
        self.head_aggregation = head_aggregation

    def set_noise_scale(
        self,
        noise_scale
    ):
        self.noise_schedule_scale = noise_scale
    
    def __call__(
        self,
        attentions,
        inputs,
        attention_mask,
        loss_mask: Optional[bool] = None,
        **kwargs
    ):
        batch_size, num_heads, num_queries, seq_len = attentions.shape
        is_self_attention = num_queries == seq_len
        # Needs fixing for self-attention probably
        unmasked_bool = ((attention_mask == 1))
        num_unmasked_queries = unmasked_bool.sum(-1)
        if self.varying_length and is_self_attention:
            num_queries_to_use = num_unmasked_queries * self.masking_ratio / self.num_per_query
        else:
            num_queries_to_use = round(seq_len * self.masking_ratio / self.num_per_query)

        attentions = nn.Softmax(dim=-1)(attentions)
        if self.head_aggregation == "mean":
            averaged_attentions = attentions.mean(dim=1)
        elif self.head_aggregation == "random_mix":
            batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_queries)
            head_idx = torch.randint(num_heads, (batch_size * num_queries, ))
            seq_idx = torch.arange(num_queries).repeat(batch_size)
            averaged_attentions = attentions[batch_idx, head_idx, seq_idx].view(batch_size, num_queries, seq_len)
        
        # Needs fixing for self-attention probably
        if self.varying_length:
            if is_self_attention:
                chosen_queries = [torch.randperm(num_unmasked_queries[i].round().long(), device=attentions.device)[:num_queries_to_use[i].round().long()] for i in range(batch_size)]
            else:
                chosen_queries = [torch.randperm(num_queries, device=attentions.device)[:num_queries_to_use] for i in range(batch_size)]
            chosen_attentions = [averaged_attentions[i, chosen_queries[i]] for i in range(batch_size)]
            summed_attentions = [i.sum(dim=0) for i in chosen_attentions]
            if self.mask_self:
                for i in range(batch_size):
                    summed_attentions[i][chosen_queries[i]] = num_heads
            if self.select_initial_ratio is not None and self.noise_schedule_scale > 0.0:
                random_ratio = self.masking_ratio * self.select_initial_ratio * self.noise_schedule_scale
                selected_ratio = self.masking_ratio * (1 - self.select_initial_ratio * self.noise_schedule_scale)
                masked_indices = [summed_attentions[i].topk(k=(selected_ratio * num_unmasked_queries[i]).round().long()).indices for i in range(batch_size)]
                unselected_bool = unmasked_bool.clone()
                for i, si in enumerate(masked_indices):
                    unselected_bool[i][si] = False
                unselected_indices = [unselected_bool[i].nonzero().squeeze(1) for i in range(batch_size)]
                random_indices = [unselected_indices[i][torch.randperm(len(unselected_indices[i]))[:(num_unmasked_queries[i] * random_ratio).round().long()]] for i in range(batch_size)]
                masked_indices = [torch.cat([masked_indices[i], random_indices[i]]) for i in range(batch_size)]
            else:
                masked_indices = [summed_attentions[i].topk(k=(self.masking_ratio * num_unmasked_queries[i]).round().long()).indices for i in range(batch_size)]
            batch_indices = torch.cat([torch.tensor(i).repeat(len(masked_indices[i])) for i in range(batch_size)])
            masked_indices = (batch_indices, torch.cat(masked_indices))
        else:
            chosen_queries = [torch.randperm(num_queries, device=attentions.device)[:num_queries_to_use] for i in range(batch_size)]
            chosen_attentions = torch.stack([averaged_attentions[i, chosen_queries[i]] for i in range(batch_size)])
            summed_attentions = chosen_attentions.sum(dim=1)
            if self.select_initial_ratio is not None and self.noise_schedule_scale > 0.0:
                select_ratio = self.masking_ratio + (self.select_initial_ratio - self.masking_ratio) * self.noise_schedule_scale
                masked_indices = summed_attentions.topk(k=round(select_ratio * seq_len), dim=-1).indices
                masked_indices = [i[torch.randperm(len(i))] for i in masked_indices]
                masked_indices = torch.stack([i[:round(self.masking_ratio * seq_len)] for i in masked_indices])
            else:
                masked_indices = summed_attentions.topk(k=round(self.masking_ratio * seq_len), dim=-1).indices
            masked_indices = (torch.repeat_interleave(torch.arange(masked_indices.shape[0]), masked_indices.shape[1]), masked_indices.flatten())
        
        if loss_mask is not None:
            if self.exclude_seen_reconstruction:
                loss_mask[masked_indices] = 1
            else:
                loss_mask[attention_mask.nonzero(as_tuple=True)] = 1

        attention_mask[masked_indices] = 0
        
        return dict(
            attentions=attentions,
            inputs=inputs,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            **kwargs,
        )


class TransformCompose(nn.Module):
    def __init__(self, transforms: List[Callable]) -> None:
        super().__init__()
        if not isinstance(transforms, list):
            raise TypeError("Argument transforms should be a list of callables")
        self.transforms = transforms

    def forward(self, **kwargs: Any) -> Any:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)

class IdentityTransform:
    def __init__(self, **kwargs):
        pass
    
    def __call__(
        self,
        **kwargs,
    ):
        if kwargs.get("loss_mask") is None:
            kwargs["loss_mask"] = kwargs.get("attention_mask")
        return dict(
            **kwargs
        )


class TransformCompose(nn.Module):
    def __init__(self, transforms: List[Callable]) -> None:
        super().__init__()
        if not isinstance(transforms, list):
            raise TypeError("Argument transforms should be a list of callables")
        self.transforms = transforms

    def forward(self, **kwargs: Any) -> Any:
        for transform in self.transforms:
            kwargs = transform(**kwargs)
        return kwargs

    def extra_repr(self) -> str:
        format_string = []
        for t in self.transforms:
            format_string.append(f"    {t}")
        return "\n".join(format_string)


TRANSFORMS2CLS = {
    "RandomlySelectedCrossAttentionMasking": RandomlySelectedCrossAttentionMasking,
    "IdentityTransform": IdentityTransform,
}

def create_transforms(transform_args):
    return TransformCompose([TRANSFORMS2CLS[c](**args) for c, args in transform_args]) if transform_args else IdentityTransform()

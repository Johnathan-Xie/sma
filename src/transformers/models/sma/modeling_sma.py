# coding=utf-8
# Copyright 2021 Deepmind and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SMA model."""

import abc
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_sma import SMAConfig
from .transforms_sma import create_transforms
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

ModalitySizeType = Mapping[str, int]
PreprocessorOutputType = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorType = Callable[..., PreprocessorOutputType]
PostprocessorType = Callable[..., Any]

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "placeholder/placeholder-model"
_CONFIG_FOR_DOC = "SMAConfig"
_TOKENIZER_FOR_DOC = "SMATokenizer"

SMA_PRETRAINED_MODEL_ARCHIVE_LIST = [

]

def cosine_scheduler(base_value, final_value, max_steps, warmup_steps=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

    iters = np.arange(max_steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def exponential_decay_scheduler(base_value, max_steps, decay_constant=None, half_ratio=0.25, patience_steps=None):
    assert (decay_constant is not None) != (half_ratio != None)
    if half_ratio is not None:
        decay_constant = math.log(half_ratio * max_steps)
    patience_schedule = np.ones((patience_steps,)) * base_value
    iters = np.arange(max_steps - patience_steps)

    decay_schedule = np.exp(iters * -decay_constant)

class CrossAttentionMaskingNoiseCallback(TrainerCallback):
    def __init__(self, resume=False, decay_type="cosine", schedule_length_ratio=None) -> None:
        super().__init__()
        self.resume = resume
        self.decay_type = decay_type
        self.schedule_length_ratio = schedule_length_ratio
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        assert model.teacher.student_model.encoder.cross_attention.attention.attention.train_noising_function is not None
        self.target = model.teacher.student_model.encoder.cross_attention.attention.attention.train_noising_function.transforms[0]

        schedule_length = round(state.max_steps * self.schedule_length_ratio) if self.schedule_length_ratio is not None else state.max_steps
        
        if self.decay_type == "exponential":
            self.schedule = exponential_decay_scheduler(1.0, schedule_length)
        elif self.decay_type == "cosine":
            self.schedule = cosine_scheduler(1.0, 0.0, schedule_length)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        noise_scale = self.schedule[state.global_step] if state.global_step < len(self.schedule) else 0.0
        self.target.set_noise_scale(noise_scale)

@dataclass
class SMAModelOutput(ModelOutput):
    """
    Base class for SMA base model's outputs, with potential hidden states, attentions and cross-attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    teacher_outputs: Optional[Dict] = None
    auxiliary_logs: Optional[Dict] = None

@dataclass
class SMADecoderOutput(ModelOutput):
    """
    Base class for SMA decoder outputs, with potential cross-attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Output of the basic decoder.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    logits: torch.FloatTensor = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class SMALatents(nn.Module):
    """Construct the latent embeddings."""

    def __init__(self, num_latents, latent_channels):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_channels))

    def forward(self, batch_size: int, valid_latents: Optional[torch.LongTensor] = None):
        latents = self.latents.expand(batch_size, -1, -1)
        return torch.gather(latents, 2, valid_latents) if valid_latents else latents

def invert_attention_mask(encoder_attention_mask: torch.Tensor, dtype) -> torch.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(dtype).min

    return encoder_extended_attention_mask

class SMASelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        bias=False,
        train_noising_args=None,
        eval_noising_args=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        self.is_cross_attention = is_cross_attention
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        # self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels, bias=bias)
        self.key = nn.Linear(kv_dim, qk_channels, bias=bias)
        self.value = nn.Linear(kv_dim, v_channels, bias=bias)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

        self.train_noising_function = create_transforms(train_noising_args) if train_noising_args is not None else None
        self.eval_noising_function = create_transforms(eval_noising_args) if eval_noising_args is not None else None
        
    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def apply_attention_noising_fn(
        self,
        attention_scores,
        attention_mask,
        inputs,
        loss_mask,
        input_ids,
    ):
        masking_outputs = None
        if self.training:
            masking_outputs = self.train_noising_function(
                attentions=attention_scores.detach(),
                attention_mask=(deepcopy(attention_mask).reshape(-1, attention_mask.shape[-1]) == 0).long(),
                inputs=inputs,
                loss_mask=loss_mask,
                input_ids=input_ids,
            )
        else:
            masking_outputs = self.eval_noising_function(
                attentions=attention_scores.detach(),
                attention_mask=(deepcopy(attention_mask).reshape(-1, attention_mask.shape[-1]) == 0).long(),
                inputs=inputs,
                loss_mask=loss_mask,
                input_ids=input_ids,
            )
        if masking_outputs is not None:
            attention_scores = attention_scores + invert_attention_mask(masking_outputs["attention_mask"], dtype=self.query.weight.dtype)
        
        return attention_scores, loss_mask
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        loss_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor]:
        hidden_states = self.layernorm1(hidden_states)
        # Only layernorm during cross attention since "inputs" is only passed during cross attention
        # queries always projected from hidden_states meaning it is always the latent (either learned or hidden embedding)
        # layernorm shouldn't be necessary since input preprocessors already have their own normalizer
        # inputs = self.layernorm2(inputs) 

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)
        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in SMAModel forward() function)
            attention_scores = attention_scores + attention_mask
            # Apply noising functions
            if (self.training and self.train_noising_function is not None) or (not self.training and self.eval_noising_function is not None):
                attention_scores, loss_mask = self.apply_attention_noising_fn(
                    attention_scores=attention_scores,
                    attention_mask=attention_mask,
                    inputs=inputs,
                    loss_mask=loss_mask,
                    input_ids=input_ids
                )
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = {
            "layer_output": context_layer,
            "attention_scores": attention_probs if output_attentions else None,
        }
        #    (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SMASelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels, bias=False):
        super().__init__()
        self.dense = nn.Linear(input_channels, output_channels, bias=bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # layernorm is applied after residual but prior to MLP
        # self.layernorm = nn.LayerNorm(config.latent_channels, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.layernorm(hidden_states)
        return hidden_states


class SMAAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
        bias=False,
        train_noising_args=None,
        eval_noising_args=None,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            qk_channels=config.qk_channels
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.attention = SMASelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            bias=bias,
            train_noising_args=train_noising_args,
            eval_noising_args=eval_noising_args,
        )
        # dense block
        output_channels = None
        if is_cross_attention:
            output_channels = self.attention.v_channels
        else:
            if output_channels is None:
                output_channels = self.attention.v_channels
        self.output = SMASelfOutput(config, input_channels=self.attention.v_channels, output_channels=output_channels, bias=bias)

        self.use_query_residual = use_query_residual
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 and not is_cross_attention else nn.Identity()

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        loss_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:

        self_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
            loss_mask=loss_mask,
            input_ids=input_ids
        )

        # Output projection
        attention_output = self.output(self_outputs["layer_output"])
        # Optionally include a residual to the original queries.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.

        # QUERY Residual always used for self attention, but has option to be disabled for cross attention
        # Note that this is the residual hidden_state from the previous MLP or the latent queries
        if self.use_query_residual:
            attention_output = self.drop_path(attention_output) + hidden_states

        outputs = self_outputs
        outputs["layer_output"] = attention_output
        #outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SMAMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, config, input_size, widening_factor, bias):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size, bias=bias)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(widening_factor * input_size, input_size, bias=bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.layernorm = nn.LayerNorm(config.latent_channels, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.layernorm(hidden_states)

        return hidden_states

class SMALayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
        train_noising_args=None,
        eval_noising_args=None,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SMAAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
            bias=config.dense_use_bias,
            train_noising_args=train_noising_args,
            eval_noising_args=eval_noising_args,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = SMAMLP(config, input_size=v_channels, widening_factor=widening_factor, bias=config.dense_use_bias)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 and not is_cross_attention else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        loss_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
            loss_mask=loss_mask,
            input_ids=input_ids,
        )
        attention_output = attention_outputs["layer_output"]

        #outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
        # ADD DROP_PATH
        layer_output = self.drop_path(layer_output) + attention_output  # residual connection

        outputs = attention_outputs

        outputs["layer_output"] = layer_output

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class SMAEmbeddingDecoder(nn.Module):
    """
    Module to decode embeddings (for discrete autoencoding)

    Args:
        config ([`SMAConfig`]):
            Model configuration.
    """

    def __init__(self, config: SMAConfig) -> None:
        super().__init__()
        self.config = config
        self.num_discrete_tokens = config.num_discrete_tokens
        self.bias = nn.Parameter(torch.zeros(self.num_discrete_tokens))

    def forward(self, hidden_states: torch.Tensor, embedding_layer: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape
        # Flatten batch dim
        output = torch.matmul(hidden_states.reshape([-1, d_model]), embedding_layer.weight.transpose(0, 1))
        output = output + self.bias

        return output.reshape([batch_size, seq_len, self.num_discrete_tokens])

class BasePreprocessor(nn.Module):

    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
        raise NotImplementedError()

class SMADiscretePreprocessor(BasePreprocessor):
    """
    Text preprocessing for SMA Encoder. Can be used to embed `inputs` and add positional encodings.

    The dimensionality of the embeddings is determined by the `d_model` attribute of the configuration.

    Args:
        config ([`SMAConfig`]):
            Model configuration.
    """

    def __init__(self, config: SMAConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(num_embeddings=config.num_discrete_tokens, embedding_dim=config.embedded_channels)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedded_channels)
        self.layernorm = nn.LayerNorm(config.embedded_channels, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    @property
    def num_channels(self) -> int:
        return self.config.embedded_channels

    def forward(
        self,
        inputs: torch.LongTensor,
    ) -> torch.FloatTensor:
        #inputs = inputs.view(inputs.shape[0], -1)
        #attention_mask = attention_mask.view(inputs.shape[0], -1)
        
        embeddings = self.embeddings(inputs)

        seq_length = inputs.shape[1]
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SMAContinuousPreprocessor(BasePreprocessor):
    """
    Continious preprocessing for SMA Encoder.

    The dimensionality of output is determined by the `d_model` attribute of the configuration.

    Args:
        config ([`SMAConfig`]):
            Model configuration.
    """

    def __init__(self, config: SMAConfig) -> None:
        super().__init__()
        self.config = config
        self.use_position_embeddings = config.use_position_embeddings
        if config.use_position_embeddings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedded_channels)
        self.input_projection = nn.Linear(config.input_channels, config.embedded_channels)
        self.layernorm = nn.LayerNorm(config.embedded_channels, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    @property
    def num_channels(self) -> int:
        return self.config.embedded_channels

    def forward(
        self,
        inputs: torch.FloatTensor,
    ) -> torch.FloatTensor:
        #inputs = inputs.view(inputs.shape[0], self.config.input_channels, -1).transpose(-1, -2)
        #attention_mask = attention_mask.view(inputs.shape[0], -1)

        seq_length = inputs.shape[1]

        embeddings = self.input_projection(inputs)
        position_ids = torch.arange(0, seq_length, device=inputs.device)
        if self.use_position_embeddings:
            embeddings = embeddings + self.position_embeddings(position_ids)
        #layernorm then dropout is intentional and is how it is done in BERT, will need ablation
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SMAPoolingDecoder(nn.Module):

    def __init__(
        self,
        config: SMAConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.linear = nn.Linear(config.latent_channels, config.output_channels)

    def forward(
        self,
        encoder_outputs,
        output_attentions: Optional[bool] = False,
    ):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        #unmasked_bool = encoder_outputs.attentions[-1].sum(1).sum(1) != 0
        encoded_latents = encoder_outputs.last_hidden_state
        averaged_latents = encoded_latents.mean(dim=1, keepdim=True)
        logits = self.linear(averaged_latents)

        return SMADecoderOutput(
            logits=logits,
        )

class SMABasicDecoder(nn.Module):
    """
    Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
    cross-attention operation, in which the latents produce keys and values.

    The shape of the output of this class depends on how one defines the output queries (also called decoder queries).

    Args:
        config ([*SMAConfig*]):
            Model configuration.
        output_num_channels (`int`, *optional*):
            The number of channels in the output. Will only be used in case *final_project* is set to `True`.
        position_encoding_type (`str`, *optional*, defaults to "trainable"):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
        output_index_dims (`int`, *optional*):
            The number of dimensions of the output queries. Ignored if 'position_encoding_type' == 'none'.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels of the decoder queries. Ignored if 'position_encoding_type' == 'none'.
        qk_channels (`int`, *optional*):
            The number of channels of the queries and keys in the cross-attention layer.
        v_channels (`int`, *optional*):
            The number of channels of the values in the cross-attention layer.
        num_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the cross-attention layer.
        widening_factor (`int`, *optional*, defaults to 1):
            The widening factor of the cross-attention layer.
        use_query_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a residual connection between the query and the output of the cross-attention layer.
        concat_preprocessed_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the preprocessed input to the query.
        final_project (`bool`, *optional*, defaults to `True`):
            Whether to project the output of the cross-attention layer to a target dimension.
        position_encoding_only (`bool`, *optional*, defaults to `False`):
            Whether to only use this class to define output queries.
    """

    def __init__(
        self,
        config: SMAConfig,
        query_num_channels: Optional[int] = 256,
        qk_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        num_heads: Optional[int] = 1,
        widening_factor: Optional[int] = 1,
        use_query_residual: Optional[bool] = False,
        num_outputs: Optional[int] = 1,
        output_channels: Optional[bool] = 2,
        encoder_input_preprocessor: Optional[nn.Module] = None,
        share_decoder_queries: Optional[bool] = False,
        share_embedding_weights: Optional[bool] = False,
        post_decoder_layers: Optional[str] = None,
        add_mask_token: Optional[bool] = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.decoding_cross_attention = SMALayer(
            config,
            is_cross_attention=True,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=query_num_channels,
            kv_dim=config.latent_channels,
            widening_factor=widening_factor,
            use_query_residual=use_query_residual,
        )
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.pre_layernorm = nn.LayerNorm(config.latent_channels)
        self.encoder_input_preprocessor = encoder_input_preprocessor

        self.learned_mask_token = None
        decoder_output_channels = v_channels
        if not share_decoder_queries or num_outputs != config.max_position_embeddings or not config.use_position_embeddings:
            self.decoder_latents = SMALatents(num_outputs, query_num_channels)
        else:
            if add_mask_token:
                self.learned_mask_token = nn.Parameter(torch.randn(self.encoder_input_preprocessor.num_channels) * self.config.initializer_range)

        if config.final_project or output_channels != decoder_output_channels:
            self.final_layernorm = nn.LayerNorm(decoder_output_channels, eps=config.layer_norm_eps)

            self.post_decoder_layers = None
            if post_decoder_layers is not None:
                layers = []
                for layer in post_decoder_layers.split(" "):
                    if layer.isdigit():
                        layer = int(layer)
                        layers.append(nn.Linear(decoder_output_channels, layer))
                        decoder_output_channels = layer
                    else:
                        layers.append(ACT2FN[layer])
                
                self.post_decoder_layers = nn.Sequential(*layers)

            if config.input_type == "discrete" and share_embedding_weights and output_channels == config.num_discrete_tokens and config.embedded_channels == decoder_output_channels:
                assert encoder_input_preprocessor
                self.embedding_layer = encoder_input_preprocessor.embeddings
                self.final_project = SMAEmbeddingDecoder(config)
                self.output_channels = config.num_discrete_tokens
            else:
                self.final_project = nn.Linear(decoder_output_channels, output_channels)
                self.output_channels = output_channels
        else:
            self.output_channels = decoder_output_channels
        
        
    def compute_decoder_mask(self, encoder_outputs):
        batch_size, num_queries, _ = encoder_outputs.last_hidden_state.shape
        masked_bool = encoder_outputs.attentions[-1].sum(1).sum(1) == 0
        decoder_mask = torch.ones(batch_size, num_queries).to(encoder_outputs.last_hidden_state.device)
        for batch_idx in range(len(masked_bool)):
            decoder_mask[batch_idx, masked_bool[batch_idx]] = 0
        return invert_attention_mask(decoder_mask, encoder_outputs.last_hidden_state.dtype)
    
    def get_decoder_queries(self, batch_size):
        if getattr(self, "decoder_latents", None):
            decoder_latents = self.decoder_latents(batch_size)
        else:
            decoder_latents = self.encoder_input_preprocessor.position_embeddings.weight.expand(batch_size, -1, -1)
        return decoder_latents

    def forward(
        self,
        encoder_outputs,
        output_attentions: Optional[bool] = False,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        encoded_latents = encoder_outputs.last_hidden_state
        batch_size, seq_len, q_head_dim = encoded_latents.shape
        cross_attentions = () if output_attentions else None
        
        decoder_queries = self.get_decoder_queries(batch_size)

        if self.learned_mask_token is not None and loss_mask is not None:
            mask_values = self.learned_mask_token.unsqueeze(0).unsqueeze(0).repeat((decoder_queries.shape[0], decoder_queries.shape[1], 1))
            mask_values[loss_mask == 0] = 0
            decoder_queries = decoder_queries + mask_values

        layer_outputs = self.decoding_cross_attention(
            decoder_queries,
            head_mask=None,
            inputs=self.pre_layernorm(encoded_latents),
            inputs_mask=self.compute_decoder_mask(encoder_outputs) if self.config.encoder_type == "self_attention" else None,
            output_attentions=output_attentions,
        )
        logits = layer_outputs["layer_output"]

        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs["attention_scores"],)
        logits = self.final_layernorm(logits)
        if self.post_decoder_layers is not None:
            logits = self.post_decoder_layers(logits)
        if getattr(self, "embedding_layer", None):
            logits = self.final_project(
                logits, embedding_layer=self.encoder_input_preprocessor.embeddings
            )
        elif self.config.final_project:
            logits = self.final_project(logits)

        return SMADecoderOutput(
            logits=logits,
            cross_attentions=cross_attentions,
        )


class SMAEncoder(nn.Module):
    """The SMA Encoder: a scalable, fully attentional encoder."""

    def __init__(self, config, cross_kv_channels=None):
        super().__init__()
        self.config = config
        # Check that we can use multihead-attention with these shapes.
        if config.latent_channels % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.latent_channels}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        if config.latent_channels % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.latent_channels}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        # Construct the cross attention layer.
        if config.encoder_type == "cross_attention":
            self.cross_attention = SMALayer(
                config,
                is_cross_attention=True,
                qk_channels=config.encoder_cross_attention_channels,
                v_channels=config.v_channels,
                num_heads=config.num_cross_attention_heads,
                q_dim=config.latent_channels,
                kv_dim=cross_kv_channels,
                widening_factor=config.cross_attention_widening_factor,
                use_query_residual=config.use_query_residual,
                train_noising_args=config.cross_train_noising_args,
                eval_noising_args=config.cross_eval_noising_args,
            )
        elif config.encoder_type == "self_attention":
            self.cross_attention = SMALayer(
                config,
                is_cross_attention=True, # current implementation is super hacky
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.latent_channels,
                kv_dim=config.embedded_channels,
                widening_factor=config.self_attention_widening_factor,
                use_query_residual=config.use_query_residual,
                train_noising_args=config.cross_train_noising_args,
                eval_noising_args=config.cross_eval_noising_args,
            )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = SMALayer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.latent_channels,
                kv_dim=config.latent_channels,
                widening_factor=config.self_attention_widening_factor,
            )
            self_attention_layers.append(layer)

        self.self_attends = nn.ModuleList(self_attention_layers)

    def build_efficient_latents(
        self,
        latents,
        attentions,
    ):
        unmasked_bool = (attentions.sum(1).sum(1) != 0)
        all_latents = [l[unmasked_bool[idx]] for idx, l in enumerate(latents)]
        longest = max(i.shape[0] for i in all_latents)

        attention_mask = torch.stack([torch.cat([torch.ones((i.shape[0], ), device=latents.device), torch.zeros(longest - i.shape[0], device=latents.device)]) for i in all_latents])
        all_latents = [torch.cat([i, torch.zeros((longest - i.shape[0], i.shape[1]), device=latents.device)]) for i in all_latents]
        all_latents = torch.stack(all_latents)
        
        return all_latents, invert_attention_mask(attention_mask, latents.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        loss_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # Apply the cross-attention between the latents (hidden_states) and inputs:
        if self.config.encoder_type == "cross_attention":
            layer_outputs = self.cross_attention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=None,
                inputs=inputs,
                inputs_mask=inputs_mask,
                output_attentions=output_attentions,
                loss_mask=loss_mask,
                input_ids=input_ids,
            )
            hidden_states = layer_outputs["layer_output"]
            
        elif self.config.encoder_type == "self_attention":
            layer_outputs = self.cross_attention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=None,
                inputs=inputs,
                inputs_mask=inputs_mask,
                output_attentions=True,
                loss_mask=loss_mask,
                input_ids=input_ids,
            )
            hidden_states, attention_mask = self.build_efficient_latents(layer_outputs["layer_output"], layer_outputs["attention_scores"])
        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs["attention_scores"],)
        
        # Apply the block of self-attention layers more than once:
        for _ in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs["layer_output"]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs["attention_scores"],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class SMAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SMAConfig
    base_model_prefix = "perceiver"
    main_input_name = "inputs"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif hasattr(module, "latents"):
            module.latents.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "position_embeddings") and (isinstance(module, SMAContinuousPreprocessor) or isinstance(module, SMADiscretePreprocessor)):
            module.position_embeddings.weight.data.normal_(mean=0.0, std=self.config.pe_initializer_range)
        elif hasattr(module, "position_embeddings") and isinstance(module, SMALatents):
            module.position_embeddings.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.ParameterDict):
            for modality in module.keys():
                module[modality].data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

class SMAModel(SMAPreTrainedModel):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config

        if config.input_type == "continuous":
            self.input_preprocessor = SMAContinuousPreprocessor(config)
        elif config.input_type == "discrete":
            self.input_preprocessor = SMADiscretePreprocessor(config)
        
        if config.encoder_type == "cross_attention":
            self.encoding_latents = SMALatents(config.num_latents, config.latent_channels)
        
        self.encoder = SMAEncoder(
            config, cross_kv_channels=self.input_preprocessor.num_channels
        )
        self.decoder = None
        if config.use_decoder:
            if config.decoder_type == "cross_attention":
                self.decoder = SMABasicDecoder(
                    config,
                    query_num_channels=config.decoder_latent_channels,
                    qk_channels=config.decoder_attention_channels,
                    v_channels=config.decoder_latent_channels,
                    num_heads=config.decoder_heads,
                    use_query_residual=True,
                    num_outputs=config.num_outputs,
                    output_channels=config.output_channels,
                    encoder_input_preprocessor=self.input_preprocessor,
                    share_decoder_queries=config.share_decoder_queries,
                    share_embedding_weights=config.share_embedding_weights,
                    post_decoder_layers=config.post_decoder_layers
                )
            elif config.decoder_type == "mean_pool":
                self.decoder = SMAPoolingDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_encoding_latents(self):
        return self.encoding_latents.latents

    def set_encoding_latents(self, value):
        self.encoding_latents.latents = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        inputs: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SMAModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        preprocessed_inputs = self.input_preprocessor(inputs)

        batch_size, seq_length, _ = preprocessed_inputs.size()
        device = preprocessed_inputs.device

        # If no attention mask is provided, make them all ones
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        # Make the attention mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = invert_attention_mask(attention_mask, preprocessed_inputs.dtype)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_blocks x num_heads]
        # and head_mask is converted to shape [num_blocks x batch x num_heads x N x N]
        head_mask = self.get_head_mask(head_mask, self.config.num_blocks * self.config.num_self_attends_per_block)
        if self.config.encoder_type == "cross_attention":
            encoding_latents = self.encoding_latents(batch_size=batch_size, valid_latents=None)

            encoder_outputs = self.encoder(
                encoding_latents,
                attention_mask=None,
                head_mask=head_mask,
                inputs=preprocessed_inputs,
                inputs_mask=extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                loss_mask=loss_mask,
                input_ids=deepcopy(inputs) if self.config.input_type == "discrete" else None,
            )
        elif self.config.encoder_type == "self_attention":
            # Weird hack where the inputs become both the inputs and "learned queries"
            # This is good since it allows for the residual connection to be applied without change
            encoder_outputs = self.encoder(
                preprocessed_inputs,
                attention_mask=None,
                head_mask=head_mask,
                inputs=preprocessed_inputs,
                inputs_mask=extended_attention_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                loss_mask=loss_mask,
                input_ids=deepcopy(inputs) if self.config.input_type == "discrete" else None,
            )

        encoder_output = encoder_outputs[0]
        logits = None
        if self.decoder:
            decoder_outputs = self.decoder(
                encoder_outputs=encoder_outputs,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits
            if output_attentions and decoder_outputs.cross_attentions is not None:
                if return_dict:
                    encoder_outputs.cross_attentions = (
                        encoder_outputs.cross_attentions + decoder_outputs.cross_attentions
                    )
                else:
                    encoder_outputs = encoder_outputs + decoder_outputs.cross_attentions
            
        if not return_dict:
            if logits is not None:
                return (logits, encoder_output) + encoder_outputs[1:]
            else:
                return (encoder_output,) + encoder_outputs[1:]

        return SMAModelOutput(
            loss=None,
            logits=logits,
            last_hidden_state=encoder_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class SMAForSequenceClassification(SMAPreTrainedModel):
    def __init__(self, config, loss_fn=None):
        config.output_channels = config.num_labels
        config.use_decoder = True
        config.cross_train_noising_args = None
        config.cross_eval_noising_args = None
        config.final_project = True
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.config = config
        self.perceiver = SMAModel(config)
        self.loss_fn = loss_fn
        self.post_init()

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SMAModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the classification/regression loss. Indices should be in `[0, ..., config.num_labels -
            1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels >
            1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import SMATokenizer, SMAForSequenceClassification

        >>> tokenizer = SMATokenizer.from_pretrained("deepmind/language-perceiver")
        >>> model = SMAForSequenceClassification.from_pretrained("deepmind/language-perceiver")

        >>> text = "hello world"
        >>> inputs = tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(inputs=inputs)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 2]
        ```"""
        if inputs is not None and input_ids is not None:
            raise ValueError("You cannot use both `inputs` and `input_ids`")
        elif inputs is None and input_ids is not None:
            inputs = input_ids
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        if labels is not None:
            if self.loss_fn is not None:
                loss = self.loss_fn(logits.squeeze(1), labels)
            else:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(labels.shape), labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.squeeze(1), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SMAModelOutput(
            loss=loss,
            logits=logits.squeeze(1),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            cross_attentions=outputs.cross_attentions if output_attentions else None,
        )
    
class SMAForSemanticSegmentation(SMAPreTrainedModel):
    def __init__(self, config):
        config.output_channels = config.num_labels
        config.use_decoder = True
        config.cross_train_noising_args = None
        config.cross_eval_noising_args = None
        config.final_project = True
        super().__init__(config)
        
        self.num_labels = config.num_labels
        self.config = config
        self.perceiver = SMAModel(config)
        self.post_init()

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SMAModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the classification/regression loss. Indices should be in `[0, ..., config.num_labels -
            1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels >
            1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import SMATokenizer, SMAForSequenceClassification

        >>> tokenizer = SMATokenizer.from_pretrained("deepmind/language-perceiver")
        >>> model = SMAForSequenceClassification.from_pretrained("deepmind/language-perceiver")

        >>> text = "hello world"
        >>> inputs = tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(inputs=inputs)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 2]
        ```"""
        if inputs is not None and input_ids is not None:
            raise ValueError("You cannot use both `inputs` and `input_ids`")
        elif inputs is None and input_ids is not None:
            inputs = input_ids
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.perceiver(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]
        #print(logits.shape)
        #print(labels.shape)
        loss = None
        if labels is not None:
            loss_mask = labels != 255
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits[loss_mask], labels[loss_mask])
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SMAModelOutput(
            loss=loss,
            logits=logits.squeeze(1),
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            cross_attentions=outputs.cross_attentions if output_attentions else None,
        )

"""
https://github.com/facebookresearch/fairseq/blob/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/modules/ema_module.py#L40
Modified from Fairseq
"""
class EMAModule(nn.Module):
    """Exponential Moving Average of Fairseq Models"""

    def __init__(
        self,
        model,
        ema_decay=0.9998,
        copy_model=True,
        device="cuda",
        skip_keys=None,
        ema_fp32=True,
    ):
        """
        @param model model to initialize the EMA with
        @param config EMAConfig object with configuration like
        ema_decay, ema_update_freq, ema_fp32
        @param device If provided, copy EMA to this device (e.g. gpu).
        Otherwise EMA is in the same device as the model.
        """
        super().__init__()
        if copy_model:
            self.model = deepcopy(model)
            self.model.requires_grad_(False)
        else:
            self.model = model

        self.decay = ema_decay
        self.fp32_params = {}
        if device is None:
            device = model.device
        print(f"Copying EMA model to device {device}")
        self.model = self.model.to(device=device)
        
        self.ema_fp32 = ema_fp32
        if ema_fp32:
            self.build_fp32_params()

    def build_fp32_params(self, state_dict=None):
        """
        Store a copy of the EMA params in fp32.
        If state dict is passed, the EMA params is copied from
        the provided state dict. Otherwise, it is copied from the
        current EMA model parameters.
        """
        if not self.ema_fp32:
            raise RuntimeError(
                "build_fp32_params should not be called if ema_fp32=False. "
                "Use ema_fp32=True if this is really intended."
            )

        if state_dict is None:
            state_dict = self.model.state_dict()

        def _to_float(t):
            return t.float() if torch.is_floating_point(t) else t

        for param_key in state_dict:
            if param_key in self.fp32_params:
                if param_key == "__sq_mom":
                    self.fp32_params[param_key] = state_dict[param_key]
                else:
                    self.fp32_params[param_key].copy_(state_dict[param_key])
            else:
                self.fp32_params[param_key] = _to_float(state_dict[param_key])
                if "__sq_mom" in self.fp32_params:
                    self.fp32_params["__sq_mom"][param_key] = torch.zeros_like(
                        self.fp32_params[param_key]
                    )

    def restore(self, state_dict, build_fp32_params=False):
        """Load data from a model spec into EMA model"""
        self.model.load_state_dict(state_dict, strict=False)
        if build_fp32_params:
            self.build_fp32_params(state_dict)

    def set_decay(self, decay, weight_decay=None):
        self.decay = decay
        if weight_decay is not None:
            self.weight_decay = weight_decay

    def get_decay(self):
        return self.decay

    def _step_internal(self, new_model):
        """One update of the EMA model based on new model weights"""
        decay = self.decay

        ema_state_dict = {}
        ema_params = (
            self.fp32_params if self.ema_fp32 else self.model.state_dict()
        )

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
                
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else deepcopy(param)
                )
                ema_params[key] = ema_param

            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue

            lr = 1 - decay
            ema_param.mul_(1 - lr)
            ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=lr)

            ema_state_dict[key] = ema_param
        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        self.restore(ema_state_dict, build_fp32_params=False)
    
    @torch.no_grad()
    def step(self, new_model):
        self._step_internal(new_model)

    def reverse(self, model):
        """
        Load the model parameters from EMA model.
        Useful for inference or fine-tuning from the EMA model.
        """
        d = self.model.state_dict()
        if "_ema" in d:
            del d["_ema"]

        model.load_state_dict(d, strict=False)
        return model

class SMABaseTeacher(nn.Module):
    """SMA base teacher with built in reconstrction_decoder"""
    def __init__(self, config, student_model):
        super().__init__()
        self.config = config
        teacher_args = config.teacher_args
        self.teacher_args = teacher_args
        self.student_model = student_model
        self.encoder_input_preprocesor = student_model.input_preprocessor

        if teacher_args.get("reconstruction_decoder_args"):
            self._create_reconstruction_decoder()

    def create_ema_teacher(self):
        self.ema_teacher = EMAModule(self.student_model)

    def create_ema_schedule(self, max_steps):
        self.ema_schedule = cosine_scheduler(
            base_value=self.teacher_args["ema_args"].get("ema_decay_start") if self.teacher_args["ema_args"].get("ema_decay_start") else 0.998,
            final_value=self.teacher_args["ema_args"].get("ema_decay_end") if self.teacher_args["ema_args"].get("ema_decay_end") else 0.9999,
            max_steps=max_steps)
        self.ema_schedule_step = 0

    def set_ema_schedule_step(self, step):
        self.ema_schedule_step = step

    def step_ema_teacher(self):
        assert self.ema_schedule_step < len(self.ema_schedule), f"EMA scheduling error. max step: {len(self.ema_schedule)}, current_step: {self.ema_schedule_step}"
        
        self.ema_teacher.set_decay(self.ema_schedule[self.ema_schedule_step])
        self.ema_teacher.step(self.student_model)
        
    def _create_reconstruction_decoder(self):
        args = self.teacher_args["reconstruction_decoder_args"]

        args["query_num_channels"] = args["query_num_channels"] if args["query_num_channels"] else self.config.embedded_channels
        args["qk_channels"] = args["qk_channels"] if args["qk_channels"] else args["query_num_channels"]
        args["v_channels"] = args["v_channels"] if args["v_channels"] else args["query_num_channels"]
        args["num_heads"] = args["num_heads"] if args["num_heads"] else self.config.decoder_heads
        args["num_outputs"] = self.config.max_position_embeddings
        args["output_channels"] = self.config.num_discrete_tokens if self.config.input_type == "discrete" else self.config.input_channels
        self.reconstruction_decoder = SMABasicDecoder(
            self.config,
            encoder_input_preprocessor=self.encoder_input_preprocesor,
            **args,
        )
    
    #def create_transforms(self):
    #    transform_args = self.teacher_args.get("transform_args")
    #    self.transforms = Compose([TRANSFORMS2CLS[c](**args) for c, args in transform_args])

    def compute_reconstruction_loss(
        self,
        encoder_outputs,
        labels: torch.Tensor,
        loss_mask: torch.Tensor = None,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        class_weightings: Optional[torch.FloatTensor] = None,
    ):
        assert getattr(self, "reconstruction_decoder")
        decoder_outputs = self.reconstruction_decoder(
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            loss_mask=loss_mask
        )

        logits = decoder_outputs.logits

        if getattr(self, "target_mean") is not None:
            labels = labels - self.target_mean.to(labels.device)
        if getattr(self, "target_std") is not None:
            labels = labels / self.target_std.to(labels.device)
        if getattr(self, "target_clip") is not None:
            labels = labels.clip(min=-self.target_clip, max=self.target_clip)
        
        if self.teacher_args["reconstruction_loss_fn"] == "crossentropy":
            if loss_mask is not None:
                logits = logits[loss_mask.nonzero(as_tuple=True)]
                labels = labels[loss_mask.nonzero(as_tuple=True)]
            if self.teacher_args.get("reconstruction_weighted_loss"):
                loss_fn = CrossEntropyLoss(reduction="none")
                loss = loss_fn(logits.view(-1, self.reconstruction_decoder.output_channels), labels.view(-1))
                
                counts = torch.zeros(self.config.num_discrete_tokens).to(labels.device)
                unique = labels.unique(return_counts=True)
                counts[unique[0]] = unique[1].float()
                class_weights = (counts.sum() / counts.clip(min=1)) / self.config.num_discrete_tokens
                
                loss = (loss * torch.index_select(class_weights, 0, labels)).mean()
            else:
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.reconstruction_decoder.output_channels), labels.view(-1))
        elif self.teacher_args["reconstruction_loss_fn"] == "mse":
            if loss_mask is not None:
                logits = logits[loss_mask.nonzero(as_tuple=True)]
                labels = labels[loss_mask.nonzero(as_tuple=True)]
            loss_fn = MSELoss()
            loss = loss_fn(logits, labels)
        return SMAModelOutput(
            loss=loss,
            cross_attentions=decoder_outputs.cross_attentions,
        )

class BYOLTeacher(SMABaseTeacher):
    def __init__(self, config, model):
        super().__init__(config, model)

    def forward(self, inputs):
        pass

class ReconstructionTeacher(SMABaseTeacher):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.train_transforms = create_transforms(self.teacher_args.get("train_transform_args"))
        self.eval_transforms = create_transforms(self.teacher_args.get("eval_transform_args"))
        if self.teacher_args.get("target_mean") is not None:
            self.target_mean = torch.Tensor(self.teacher_args.get("target_mean"))
        else:
            self.target_mean = None
        
        if self.teacher_args.get("target_std") is not None:
            self.target_std = torch.Tensor(self.teacher_args.get("target_std"))
        else:
            self.target_std = None
        
        self.target_clip = self.teacher_args.get("target_clip")
    
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_loss: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SMAModelOutput]:
        if self.training:
            augment_outputs = self.train_transforms(
                inputs=inputs,
                attention_mask=attention_mask,
                head_mask=head_mask,
                loss_mask=torch.zeros_like(attention_mask) if loss_mask is None else loss_mask,
            )
        else:
            augment_outputs = self.eval_transforms(
                inputs=inputs,
                attention_mask=attention_mask,
                head_mask=head_mask,
                loss_mask=torch.zeros_like(attention_mask) if loss_mask is None else loss_mask,
            )
        
        student_outputs = self.student_model(
            inputs=augment_outputs["inputs"],
            attention_mask=augment_outputs["attention_mask"],
            head_mask=augment_outputs["head_mask"],
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            loss_mask=augment_outputs["loss_mask"],
        )
        
        if self.teacher_args.get("reconstruction_decoder_args"):
            reconstruction_outputs = self.compute_reconstruction_loss(
                encoder_outputs=student_outputs,
                labels=inputs,
                loss_mask=augment_outputs["loss_mask"],
                output_attentions=output_attentions,
            )
            reconstruction_loss = reconstruction_outputs.loss
        if output_attentions:
            all_cross_attentions = student_outputs.cross_attentions + (reconstruction_outputs.cross_attentions)
        return SMAModelOutput(
            loss=reconstruction_loss if return_loss else None,
            logits=getattr(student_outputs, "logits"),
            last_hidden_state=getattr(student_outputs, "last_hidden_state"),
            hidden_states=getattr(student_outputs, "hidden_states"),
            attentions=getattr(student_outputs, "attentions"),
            cross_attentions=all_cross_attentions if output_attentions else None,
        )

TEACHER2CLS = {
    "ReconstructionTeacher": ReconstructionTeacher,
}

class SMAForSSL(SMAPreTrainedModel):
    """
    DAP Module for Self-Supervised Learning. Mainly just an interface for saving and loading models.
    main computation for SSL happens within teachers
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.perceiver = SMAModel(config)
        self.teacher = TEACHER2CLS[config.teacher_name](config, self.perceiver)
        
        self.post_init()
    
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_loss: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        input_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SMAModelOutput]:
        if inputs is not None and input_ids is not None:
            raise ValueError("You cannot use both `inputs` and `input_ids`")
        elif inputs is None and input_ids is not None:
            inputs = input_ids
        return self.teacher(
            inputs=inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            loss_mask=loss_mask,
        )

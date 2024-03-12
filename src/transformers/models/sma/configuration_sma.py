# coding=utf-8
# Copyright Deepmind and The HuggingFace Inc. team. All rights reserved.
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
""" SMA model configuration"""

from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import FeatureExtractionMixin
from ...onnx import OnnxConfig
from ...onnx.utils import compute_effective_axis_dimension
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)

SMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    
}


class SMAConfig(PretrainedConfig):
    model_type = "sma"

    def __init__(
        self,
        use_position_embeddings=True,
        num_latents=512,
        latent_channels=1024,
        embedded_channels=256,
        num_blocks=1,
        num_self_attends_per_block=6,
        num_self_attention_heads=8,
        num_cross_attention_heads=1,
        qk_channels=None,
        v_channels=None,
        dense_use_bias=False,
        encoder_type="cross_attention",
        encoder_cross_attention_channels=None,
        cross_train_noising_args=None,
        cross_eval_noising_args=None,
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.0,
        initializer_range=0.02,
        pe_initializer_range=0.02,
        layernorm_eps=1e-12,
        use_query_residual=True,
        max_position_embeddings=50176,
        input_channels=3,
        input_type="continuous",
        project_after_concat=True,
        num_discrete_tokens=262,
        loss_fn="mse",
        use_decoder=False,
        decoder_type="cross_attention",
        decoder_latent_channels=None,
        decoder_attention_channels=None,
        decoder_heads=1,
        post_decoder_layers=None,
        final_project=True,
        output_channels=3,
        num_outputs=1,
        share_decoder_queries=False,
        share_embedding_weights=False,
        teacher_name="ReconstructionTeacher",
        teacher_args={
        },
        **kwargs
    ):
        super().__init__(**kwargs)

        self.use_position_embeddings = use_position_embeddings
        self.num_latents = num_latents
        self.latent_channels = latent_channels
        self.embedded_channels = embedded_channels
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.cross_train_noising_args = cross_train_noising_args
        self.cross_eval_noising_args = cross_eval_noising_args
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.dense_use_bias = dense_use_bias
        
        self.encoder_type = encoder_type
        self.encoder_cross_attention_channels = encoder_cross_attention_channels if encoder_cross_attention_channels else qk_channels
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.hidden_act = hidden_act
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.pe_initializer_range = pe_initializer_range
        self.layernorm_eps = layernorm_eps
        self.use_query_residual = use_query_residual
        
        self.input_type = input_type
        self.input_channels = input_channels
        self.max_position_embeddings = max_position_embeddings
        self.num_discrete_tokens = num_discrete_tokens

        self.loss_fn = loss_fn
        self.project_after_concat = project_after_concat

        self.use_decoder = use_decoder
        self.decoder_latent_channels = decoder_latent_channels if decoder_latent_channels else latent_channels
        self.decoder_heads = decoder_heads
        self.share_decoder_queries = share_decoder_queries
        self.share_embedding_weights = share_embedding_weights
        self.post_decoder_layers = post_decoder_layers
        self.final_project = final_project

        self.decoder_type = decoder_type
        self.decoder_attention_channels = decoder_attention_channels
        
        self.output_channels = output_channels if output_channels else input_channels
        self.num_outputs = num_outputs if num_outputs else max_position_embeddings
        self.layer_norm_eps = 1e-12

        self.teacher_name = teacher_name
        self.teacher_args = teacher_args
        
class SMAOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("inputs", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
    ) -> Mapping[str, Any]:
        # copied from `transformers.onnx.config.OnnxConfig` and slightly altered/simplified

        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(
                batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
            )
            # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(
                seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
            )
            # Generate dummy inputs according to compute batch and sequence
            dummy_input = [" ".join(["a"]) * seq_length] * batch_size
            inputs = dict(preprocessor(dummy_input, return_tensors=framework))
            inputs["inputs"] = inputs.pop("input_ids")
            return inputs
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            inputs = dict(preprocessor(images=dummy_input, return_tensors=framework))
            inputs["inputs"] = inputs.pop("pixel_values")
            return inputs
        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

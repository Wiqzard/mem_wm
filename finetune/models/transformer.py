from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers import ConfigMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph


from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps

from einops import rearrange, repeat
from finetune.models.action_encoder import ActionEncoder


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.ff_inner_dim = ff_inner_dim
        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)
        if isinstance(ff_output, tuple):
            ff_output, routing_weights = ff_output
        else:
            routing_weights = None

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states, routing_weights


class CogVideoXTransformer3DActionModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        self.action_encoder = ActionEncoder(hidden_dim=128, out_dim=text_embed_dim, inner_dim=num_attention_heads*attention_head_dim, group_size=4)
        #self.action_encoder = ActionEncoder(hidden_dim=128, out_dim=text_embed_dim, inner_dim=num_attention_heads*attention_head_dim, group_size=4)
        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)
        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs
        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)
        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels
        self.proj_out = nn.Linear(inner_dim, output_dim)
        self.gradient_checkpointing = False

    def encode_actions(self, actions: Dict[str, Any], uc=False, device=None, dtype=None, cfg=False, mask_ratio=0.0, sequence_length=226):
        B, T = actions["dx"].shape
        actions = {k: v.to(device, dtype=dtype) for k, v in actions.items()}
        if cfg:
            encoded_uc_actions = self.action_encoder(actions, uc=True, sequence_length=sequence_length)
            encoded_actions = self.action_encoder(actions, uc=False, sequence_length=sequence_length)
            encoded_actions = torch.cat([encoded_actions, encoded_uc_actions], dim=0)
            return encoded_actions

        if uc:
            encoded_actions = self.action_encoder(actions, uc=True,  sequence_length=sequence_length)
        else:
            encoded_actions = self.action_encoder(actions, uc=False, mask_ratio=mask_ratio,  sequence_length=sequence_length)

        return encoded_actions

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        actions: Optional[torch.Tensor] = None,
        cfg: Optional[Dict[str, Any]] = False,
        uc: bool = False,
        mask_ratio: float = 0.0,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        batch_size, num_frames, channels, height, width = hidden_states.shape
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        encoded_actions = self.encode_actions(
           actions, uc=uc, device=hidden_states.device, dtype=hidden_states.dtype, cfg=cfg, mask_ratio=mask_ratio
        )

        if False:
            encoded_actions, continuous_actions  = self.encode_actions(
                actions, uc=uc, device=hidden_states.device, dtype=hidden_states.dtype, cfg=cfg, mask_ratio=mask_ratio
            )

        if encoder_hidden_states is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoded_actions], dim=1)
            assert encoder_hidden_states.shape[1] == self.max_text_seq_length
        else:
            encoder_hidden_states = encoded_actions

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if False:
            hidden_states = rearrange(hidden_states, "b (t s) c -> b t s c", t=num_frames) 
            continuous_actions = repeat(continuous_actions, 'b t c -> b t s c', s=hidden_states.shape[2])
            hidden_states[:, 1:, :, :] = hidden_states[:, 1:, :, :] + continuous_actions
            hidden_states = rearrange(hidden_states, "b t s c -> b (t s) c")
        
        routing_layer_weights = {}

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:


                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                block_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                block_outputs = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            if block_outputs[2] is not None:
                hidden_states, encoder_hidden_states, routing_weights = block_outputs
                routing_layer_weights[f"block_{i}"] = routing_weights
            else:
                hidden_states, encoder_hidden_states, _ = block_outputs

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]
        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        if p_t is None:
            output = hidden_states.reshape(
                batch_size, num_frames, height // p, width // p, -1, p, p
            )
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = (
                output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
            )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        if not return_dict:
            if len(routing_layer_weights) > 0:
                return output, routing_layer_weights

            return (output,)
        return Transformer2DModelOutput(sample=output)


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.
        <Tip warning={true}>
        This API is 🧪 experimental.
        </Tip>
        """
        self.original_attn_processors = None
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )
        self.original_attn_processors = self.attn_processors
        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)
        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.
        <Tip warning={true}>
        This API is 🧪 experimental.
        </Tip>
        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

config_5b = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 32,
    "max_text_seq_length": 226,
    "norm_elementwise_affine": True,
    "norm_eps": 1e-05,
    "num_attention_heads": 48,
    "num_layers": 42,
    "ofs_embed_dim": 512,
    "out_channels": 16,
    "patch_bias": False,
    "patch_size": 2,
    "patch_size_t": 2,
    "sample_frames": 81,
    "sample_height": 300,
    "sample_width": 300,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    "text_embed_dim": 4096,
    #"text_embed_dim": 1920,
    "time_embed_dim": 512,
    "timestep_activation_fn": "silu",
    "use_learned_positional_embeddings": False,
    "use_rotary_positional_embeddings": True,
}


config_2b_iv = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 32,
    "max_text_seq_length": 226,
    "norm_elementwise_affine": True,
    "norm_eps": 1e-05,
    "num_attention_heads": 30,
    "num_layers": 30,
    "out_channels": 16,
    "patch_size": 2,
    "sample_frames": 49,
    "sample_height": 60,
    "sample_width": 90,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    #"text_embed_dim": 1920,
    "text_embed_dim": 4096,
    "time_embed_dim": 512,
    "timestep_activation_fn": "silu",
    "use_rotary_positional_embeddings": False,
}
config_2b = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 16,
    "max_text_seq_length": 226,
    "norm_elementwise_affine": True,
    "norm_eps": 1e-05,
    "num_attention_heads": 30,
    "num_layers": 30,
    "out_channels": 16,
    "patch_size": 2,
    "sample_frames": 49,
    "sample_height": 60,
    "sample_width": 90,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    "text_embed_dim": 4096,
    "time_embed_dim": 512,
    "timestep_activation_fn": "silu",
    "use_rotary_positional_embeddings": False,
}

config_100m = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "num_attention_heads": 8, #12
    "num_layers": 12,         
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "use_rotary_positional_embeddings": False,
    "in_channels": 32,
    "out_channels": 16,
    "patch_size": 2,
    "max_text_seq_length": 226,
    "text_embed_dim": 768,     
    "time_embed_dim": 256,     
    "norm_elementwise_affine": True,
    "norm_eps": 1e-5,
    "sample_frames": 49,
    "sample_height": 60,
    "sample_width": 90,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    "timestep_activation_fn": "silu"
}

config_50m = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "num_layers": 12,         
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "use_rotary_positional_embeddings": False,
    "in_channels": 32,
    "out_channels": 16,
    "patch_size": 2,
    "max_text_seq_length": 226,
    "text_embed_dim": 768,     
    "time_embed_dim": 256,     
    "norm_elementwise_affine": True,
    "norm_eps": 1e-5,
    "sample_frames": 49,
    "sample_height": 60,
    "sample_width": 90,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    "timestep_activation_fn": "silu"
}


if __name__ == "__main__":
    def print_trainable_parameters(model: nn.Module) -> None:
        """
        Prints out the names, shapes, and parameter counts for all parameters in `model` that are trainable.
        Also prints a summary of the total number of trainable parameters vs. total parameters.
        
        Args:
            model (nn.Module): The model whose parameters will be printed.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)\n")
        
        print("Trainable parameter details:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Convert the shape tuple to a string so that it can be formatted with a width specifier.
                shape_str = str(tuple(param.shape))
                print(f" - {name:40s} | Shape: {shape_str:20s} | Count: {param.numel():,}")

            
            



    from custom_moe2 import MixLoraConfig, MoeLoraLayer, inject_adapter_in_model, mem_config, save_adapter_weights, load_adapter_weights , set_adapter_trainable, fix_routing_weights, disable_adapter, activate_adapter
    with torch.no_grad(), torch.cuda.amp.autocast():
        model = CogVideoXTransformer3DActionModel(**config_2b_iv).cuda().to(dtype=torch.bfloat16)
        #inject_adapter_in_model(model, mem_config, weights=None, device="cuda", dtype=torch.bfloat16)
        #save_adapter_weights(model, save_directory="mix_lora_weights")
        load_adapter_weights(model=model, load_directory="mix_lora_weights", device="cuda")
        disable_adapter(model)

        model.cuda()

        #fix_routing_weights(model=model, routing_weights=torch.zeros((4), device="cuda"))
        #print_trainable_parameters(model) 
        #set_adapter_trainable(model=model)
        #print_trainable_parameters(model) 

        #print(model)
        print(model)
        actions = model.action_encoder.get_dummy_input(batch_size=1, num_frames=48)
        actions = {k: v.cuda() for k, v in actions.items()}
        test_input = torch.randn(1, 49, 32, 20, 30).cuda()
        test_encoder_hidden_states =  None # torch.randn(1, 226, 16, 60, 90).cuda()
        test_timestep = torch.tensor([0]).cuda()
        output = model(test_input, test_encoder_hidden_states, test_timestep, actions=actions, return_dict=False)
        print(output.sample.shape)
#    with torch.no_grad(), torch.cuda.amp.autocast():
#        model = CogVideoXTransformer3DActionModel(**config_2b_iv).cuda().to(dtype=torch.bfloat16)
#        print(model)
#        actions = model.action_encoder.get_dummy_input(batch_size=1, num_frames=48)
#        actions = {k: v.cuda() for k, v in actions.items()}
#        test_input = torch.randn(1, 49, 32, 20, 30).cuda()
#        test_encoder_hidden_states =  None # torch.randn(1, 226, 16, 60, 90).cuda()
#        test_timestep = torch.tensor([0]).cuda()
#        output = model(test_input, test_encoder_hidden_states, test_timestep, actions=actions)
#        print(output.sample.shape)
#
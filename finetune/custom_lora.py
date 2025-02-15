import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List
import importlib.metadata
import importlib.util
import logging
import math
import copy
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN

from packaging import version


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_package_available(
    pkg_name: str, pkg_version: Optional[str] = None
) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if pkg_version is not None:
        return package_exists and version.parse(package_version) >= version.parse(pkg_version)
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


@dataclass
class AdapterConfig:
    base_model_: str = None
    task_type_: str = None
    peft_type_: str = None
    adapter_name_: str = None
    model_type_: str = None
    dtype_: torch.dtype = None

    @property
    def base_model_name_or_path(self):
        return self.base_model_

    @property
    def adapter_name(self):
        return self.adapter_name_

    def check(self) -> "AdapterConfig":
        assert isinstance(self.base_model_, str)
        assert isinstance(self.task_type_, str)
        assert isinstance(self.peft_type_, str)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            base_model_=config["base_model_name_or_path"],
            task_type_=config["task_type"],
            peft_type_=config["peft_type"],
        )

    def export(self) -> Dict[str, any]:
        config = {}
        config["bias"] = "none"
        config["peft_type"] = self.peft_type_
        config["task_type"] = self.task_type_
        config["base_model_name_or_path"] = self.base_model_

        return config


lora_target_modules = {
    # LLaMA names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
    # Phi names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "dense": False,
    "fc1": False,
    "fc2": False,
    # Phi3 names
    "qkv_proj": False,
    "o_proj": False,
    "gate_up_proj": False,
    "down_proj": False,
}


@dataclass
class LoraConfig(AdapterConfig):
    # Weight-Decomposed Low-Rank Adaptation
    use_dora_: bool = False
    # Rank-Stabilized LoRA
    # sets the adapter scaling factor to `alpha/math.sqrt(r)`
    use_rslora_: bool = False
    # can be original or gaussian
    lora_init_: str = "original"
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None

    def check(self) -> "LoraConfig":
        super().check()
        assert isinstance(self.use_dora_, bool)
        assert isinstance(self.use_rslora_, bool)
        assert isinstance(self.lora_init_, str) and self.lora_init_ in [
            "original",
            "gaussian",
        ]
        # assert isinstance(self.lora_r_, int) and self.lora_r_ > 0
        assert isinstance(self.lora_alpha_, int) and self.lora_alpha_ > 0
        assert isinstance(self.lora_dropout_, float) and self.lora_dropout_ >= 0
        assert isinstance(self.target_modules_, Dict)
        for key, value in self.target_modules_.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(value, bool)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.use_dora_ = config.get("use_dora", False)
        lora_config.use_rslora_ = config.get("use_rslora", False)
        lora_config.lora_init_ = config.get("lora_init", "original")
        lora_config.lora_r_ = config["r"]
        lora_config.lora_alpha_ = config["lora_alpha"]
        lora_config.lora_dropout_ = config["lora_dropout"]
        lora_config.target_modules_ = copy.deepcopy(lora_target_modules)
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        if self.use_dora_:
            config["use_dora"] = True
        if self.use_rslora_:
            config["use_rslora"] = True
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        tgt_list = []
        for target, value in self.target_modules_.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list

        return config


available_routing_strategies = ["mixlora"]


@dataclass
class MixLoraConfig(LoraConfig):
    # expert lora
    expert_config_: LoraConfig = None
    # router config
    router_aux_loss_coef_: float = None
    router_init_range_: float = None
    routing_strategy_: str = None
    jitter_noise_: float = None
    router_loss_: bool = True
    num_experts_: int = None
    act_fn_: Optional[Union[str, torch.nn.Module]] = None
    top_k_: int = None

    def check(self) -> "MixLoraConfig":
        super().check()
        if self.expert_config_ is not None:
            self.expert_config_.check()
        assert isinstance(self.router_aux_loss_coef_, float) and self.router_aux_loss_coef_ >= 0
        assert isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        assert (
            isinstance(self.routing_strategy_, str)
            and self.routing_strategy_ in available_routing_strategies
        )
        assert isinstance(self.jitter_noise_, float) and self.jitter_noise_ >= 0
        assert isinstance(self.router_loss_, bool)
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert self.act_fn_ is None or (isinstance(self.act_fn_, str) and self.act_fn_ in ACT2FN)
        if self.routing_strategy_ == "mixlora":
            assert isinstance(self.top_k_, int) and self.top_k_ > 0
        else:
            raise NotImplementedError()

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MixLoraConfig":
        lora_config = MixLoraConfig(**LoraConfig.from_config(config).__dict__)
        lora_config.routing_strategy_ = config.get("routing_strategy", None)
        assert (
            lora_config.peft_type_ == "MIXLORA"
            and lora_config.routing_strategy_ is not None
            and lora_config.routing_strategy_ == "mixlora"
        ), "MixLoraConfig only supports MixLoRA models with 'mixlora' routing_strategy."
        if "expert_lora" in config:
            expert_config = copy.deepcopy(config)
            expert_config.update(config["expert_lora"])
            lora_config.expert_config_ = LoraConfig().from_config(expert_config)
        lora_config.router_aux_loss_coef_ = config.get(
            "router_aux_loss_coef", 0.001
        )  # for training
        lora_config.router_loss_ = config.get("router_loss", True)
        lora_config.num_experts_ = config["num_experts"]
        # left blank to automatically use the original act_fn of FFN
        lora_config.act_fn_ = config.get("act_fn", None)
        if lora_config.routing_strategy_ == "mixlora":
            lora_config.router_init_range_ = config.get("router_init_range", 0.02)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.0)
            lora_config.top_k_ = config.get("top_k", 2)
        else:
            raise NotImplementedError()

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MIXLORA"
        if self.expert_config_ is not None:
            expert_config = self.expert_config_.export()
            expert_config.pop("peft_type")
            expert_config.pop("target_modules")
            config["expert_lora"] = expert_config
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        if self.act_fn_ is not None and isinstance(self.act_fn_, str):
            config["act_fn"] = self.act_fn_
        if self.routing_strategy_ == "mixlora":
            config["top_k"] = self.top_k_
        else:
            raise NotImplementedError()

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        if self.expert_config_ is None:
            config = copy.deepcopy(super())
        else:
            config = copy.deepcopy(self.expert_config_)
        return config


# def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
#    # BNB requires CUDA weights
#    device = weight.device
#    is_cpu = device.type == torch.device("cpu").type
#    if is_cpu:
#        weight = weight.to(torch.device("cuda"))
#
#    cls_name = weight.__class__.__name__
#    if cls_name == "Params4bit":
#        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
#        if is_cpu:
#            dequantized = dequantized.to(device)
#        return dequantized
#
#    if state.SCB is None:
#        state.SCB = weight.SCB
#
#    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
#    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
#    im, Sim = bnb.functional.transform(im, "col32")
#    if state.CxB is None:
#        state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
#    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
#    dequantized = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()
#    if is_cpu:
#        dequantized = dequantized.to(device)
#    return dequantized
#
#
# def dequantize_module_weight(module: torch.nn.Module) -> torch.nn.Parameter:
#    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
#        weight = module.dequantize()
#        return weight
#
#    weight = module.weight
#    if not isinstance(weight, torch.nn.Parameter):
#        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")
#
#    cls_name = weight.__class__.__name__
#    if cls_name not in ("Params4bit", "Int8Params"):
#        return weight
#
#    quant_state = getattr(module, "state", None)
#    device = weight.device
#    is_cpu = device.type == torch.device("cpu").type
#    weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
#    if is_cpu:
#        # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
#        module.weight = module.weight.to(device)
#    return weight


class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        config: LoraConfig,
        lora_r: int = None,
        weight: Tuple[torch.Tensor, torch.Tensor] = (None, None),
        device: str = None,
    ):
        super().__init__()

        # if not isinstance(base_layer, nn.Linear):
        #    assert isinstance(base_layer, Linear8bitLt) or isinstance(
        #        base_layer, Linear4bit
        #    ), f"Unsupported base layer type '{type(base_layer)}'."

        if isinstance(base_layer, Linear4bit):
            out_dim, in_dim = (
                base_layer.out_features,
                base_layer.in_features,
            )
        else:
            out_dim, in_dim = base_layer.weight.shape

        self.base_layer_ = base_layer
        self.device_ = torch.device(device) if device else base_layer.weight.device
        self.dtype_ = config.dtype_

        self.initializer_ = config.lora_init_
        self.r_ = config.lora_r_ if lora_r is None else lora_r
        print(self.r_)
        self.alpha_ = config.lora_alpha_

        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        self.in_features_ = in_dim
        self.out_features_ = out_dim

        assert config.lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=config.lora_dropout_)

        self.lora_A = nn.Linear(
            self.in_features_,
            self.r_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )
        self.lora_B = nn.Linear(
            self.r_,
            self.out_features_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )

        self.use_dora_: bool = config.use_dora_
        self.magnitude_vector_: nn.Parameter = None

        self.reset_parameters(weight)

    def _get_weight_norm(self) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = self.base_layer_.module.to(
            self.dtype_
        )  # dequantize_module_weight(self.base_layer_).to(self.dtype_)
        lora_weight = self.lora_B.weight @ self.lora_A.weight
        weight = weight + self.scaling_ * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def reset_parameters(self, weight: Tuple[torch.Tensor, torch.Tensor] = (None, None)) -> None:
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(weight, Tuple)
        assert len(weight) == 2
        assert ((weight[0] is None) and (weight[1] is None)) or (
            isinstance(weight[0], torch.Tensor) and isinstance(weight[1], torch.Tensor)
        )

        if weight == (None, None):
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_A.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_B.weight)
        else:
            with torch.no_grad():
                self.lora_A.weight.copy_(weight[0])
                self.lora_B.weight.copy_(weight[1])

        if self.use_dora_:
            self.magnitude_vector_ = nn.Parameter(self._get_weight_norm(), requires_grad=True)

    def apply_dora(
        self,
        residual: torch.Tensor,
        result_lora: torch.Tensor,
    ):
        weight_norm = self._get_weight_norm().detach()
        mag_norm_scale = (self.magnitude_vector_ / weight_norm).view(1, -1)
        return mag_norm_scale * residual + mag_norm_scale * result_lora

    def lora_forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        result_lora = (
            self.lora_B(self.lora_A(self.dropout_(hidden_states.to(self.dtype_)))) * self.scaling_
        )
        if self.use_dora_:
            return self.apply_dora(residual, result_lora).to(hidden_states.dtype)
        else:
            return residual + result_lora.to(residual.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.base_layer_(hidden_states)
        return self.lora_forward(residual, hidden_states)


def _slice_tensor(
    data: torch.Tensor,
    slice: torch.Tensor,
    dtype: torch.dtype,
    last_value: Optional[torch.Tensor] = None,
):
    if last_value is None:
        # for macOS debugging, please uncomment this line
        # assert data.dtype in (torch.float, torch.int, torch.bool)
        return data[None, slice].reshape(-1, data.shape[-1]).to(dtype)
    else:
        return last_value


_compatible_model_types = {
    "llama": "_llama_forward",
    "gemma": "_llama_forward",
    "gemma2": "_llama_forward",
    "qwen2": "_llama_forward",
    "mistral": "_llama_forward",
    "phi": "_phi_forward",
    "phi3": "_phi3_forward",
}

class MixLoraSparseMoe(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        config: MixLoraConfig,
    ) -> None:
        super().__init__()

        self.dtype_: torch.dtype = config.dtype_
        self.gate_: nn.Parameter = None

        # This is the original MLP (or FF) block
        self.base_layer_ = base_layer

        # IMPORTANT: use nn.ModuleDict instead of a plain dict
        self.experts_ = nn.ModuleDict()

        self.act_fn_ = (
            ACT2FN[config.act_fn_] if isinstance(config.act_fn_, str) else config.act_fn_
        )
        self.num_experts_: int = config.num_experts_
        self.topk_: int = config.top_k_
        self.jitter_noise_: float = config.jitter_noise_

    def set_gate(self, in_features: int, router_init_range: float = 0.02):
        # Properly register this as a parameter
        self.gate_ = nn.Parameter(
            torch.empty(self.num_experts_, in_features, dtype=self.dtype_)
        )
        nn.init.normal_(self.gate_, mean=0.0, std=router_init_range)

    def forward_cog(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        # (A) Base forward for "common_fc1":
        common_fc1 = self.base_layer_.net[0](hidden_states.to(input_dtype))
        common_act = self.act_fn_(common_fc1)

        # We'll produce final hidden states for each expert separately.
        final_expert_states = []

        # (B) For each expert, gather tokens that route to that expert
        for expert_idx in range(self.num_experts_):
            # Indices for tokens that go to this expert
            _, top_positions = torch.where(expert_mask[expert_idx])

            # Retrieve the LoraLinear for fc1 and fc2:
            lora_fc1 = self.experts_.get(f"experts.{expert_idx}.net.0", None)
            lora_fc2 = self.experts_.get(f"experts.{expert_idx}.net.2", None)

            # (B.1) If we have a LoRA for net[0], apply it:
            if lora_fc1 is not None:
                sub_input = common_fc1[top_positions].to(input_dtype)
                sub_original = hidden_states[top_positions].to(input_dtype)
                fc1_expert = lora_fc1.lora_forward(sub_input, sub_original)
                fc1_expert = self.act_fn_(fc1_expert)
            else:
                fc1_expert = common_act[top_positions].to(input_dtype)

            # (B.2) Second linear (net[2]) from the base:
            fc2_base = self.base_layer_.net[2](fc1_expert)

            if lora_fc2 is not None:
                fc2_expert = lora_fc2.lora_forward(fc2_base, fc1_expert)
            else:
                fc2_expert = fc2_base

            final_expert_states.append(fc2_expert)

        return final_expert_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.jitter_noise_ > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_
            )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = F.linear(hidden_states, self.gate_.to(hidden_states))
        routing_weights = F.softmax(router_logits, dim=1, dtype=self.dtype_)

        # top-k gating
        routing_weights, selected_experts = torch.topk(routing_weights, self.topk_, dim=-1)
        # re-normalize
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        # One-hot encode the selected experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts_)
        # shape => (batch*seq, topk, num_experts) => we want (num_experts, batch*seq)
        # We'll do a small reshape trick:
        #   selected_experts: (batch*seq, topk)
        #   => one_hot => (batch*seq, topk, num_experts)
        # We want (num_experts, batch*seq).
        # However, if top_k_ > 1, we'd handle it differently. This code
        # merges them, which might be OK for demonstration.

        # For the forward pass in forward_cog, you used expert_mask of shape [n_experts, (batch*seq)].
        # We'll flatten out top_k dimension:
        expert_mask = expert_mask.sum(dim=1).T  # shape => (num_experts, batch*seq)

        # forward pass on each expert
        expert_states = self.forward_cog(
            expert_mask,
            hidden_states,
            input_dtype,
        )

        # Combine
        for expert_idx in range(self.num_experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_hidden_states = (
                expert_states[expert_idx] * routing_weights[top_x, idx, None]
            )
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(self.dtype_))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)

        return final_hidden_states, routing_weights


def _inject_attn_module(
    layer_idx: int,
    self_attn: nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    """
    Example injection for attention modules, storing them directly
    via `setattr(...)` so PyTorch tracks them.
    """
    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(self_attn, proj_name):
            continue
        base_layer = getattr(self_attn, proj_name)
        layer_prefix_name = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"

        # Replace the original linear with a LoraLinear
        setattr(
            self_attn,
            proj_name,
            LoraLinear(
                base_layer,
                config,
                (
                    weights.get(f"{layer_prefix_name}.lora_A.weight", None),
                    weights.get(f"{layer_prefix_name}.lora_B.weight", None),
                ),
            ),
        )

def _inject_mlp_module(
    layer_idx: int,
    mlp: nn.Module,
    config: MixLoraConfig,
    weights: Optional[Dict[str, torch.Tensor]],
):
    """
    We replace mlp with our MixLoraSparseMoe so it handles gating + LoRA
    injection for each expert.
    """
    moe_layer = MixLoraSparseMoe(mlp, config)
    if weights is None:
        moe_layer.set_gate(in_features=1920)  # e.g. '1920' must match your hidden size
    else:
        gate_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
        moe_layer.gate_ = nn.Parameter(weights[gate_key].to(config.dtype_))

    # Attach our new moe_layer as the .mixlora_moes for reference
    if not hasattr(mlp, "mixlora_moes"):
        mlp.mixlora_moes = {}
    mlp.mixlora_moes[config.adapter_name_] = moe_layer

    # We override the mlp forward with the MixLoraSparseMoe forward
    mlp.forward = moe_layer.forward

    base_layer_1 = mlp.net[0].proj
    base_layer_2 = mlp.net[2]

    # Suppose config.lora_r_ can be a single int or a list of ranks
    lora_ranks = config.lora_r_
    if isinstance(lora_ranks, int):
        lora_ranks = [lora_ranks] * config.num_experts_
    assert len(lora_ranks) == config.num_experts_, (
        f"Expected {config.num_experts_} ranks, got {len(lora_ranks)}"
    )

    for expert_idx in range(config.num_experts_):
        layer_1_prefix_name = f"mixlora.layers.{layer_idx}.mlp.net.0.experts.{expert_idx}"
        layer_2_prefix_name = f"mixlora.layers.{layer_idx}.mlp.net.2.experts.{expert_idx}"

        # Register the LoRA for net.0
        moe_layer.experts_[f"experts.{expert_idx}.net.0"] = LoraLinear(
            base_layer_1,
            config,
            lora_ranks[expert_idx],
            (
                weights.get(f"{layer_1_prefix_name}.lora_A.weight", None) if weights is not None else None,
                weights.get(f"{layer_1_prefix_name}.lora_B.weight", None) if weights is not None else None,
            )
        )
        # Register the LoRA for net.2
        moe_layer.experts_[f"experts.{expert_idx}.net.2"] = LoraLinear(
            base_layer_2,
            config,
            lora_ranks[expert_idx],
            (
                weights.get(f"{layer_2_prefix_name}.lora_A.weight", None) if weights is not None else None,
                weights.get(f"{layer_2_prefix_name}.lora_B.weight", None) if weights is not None else None,
            )
        )

    # Optionally inject additional submodules
    targets = config.target_modules_.copy()
    if "net" in targets:
        # we handled net[0]/net[2] ourselves
        del targets["net"]

    for proj_name, inject in targets.items():
        if not inject or not hasattr(mlp, proj_name):
            continue
        base_layer = getattr(mlp, proj_name)
        for expert_idx in range(config.num_experts_):
            layer_prefix_name = (
                f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
            )
            moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = LoraLinear(
                base_layer,
                config,
                (
                    weights.get(f"{layer_prefix_name}.lora_A.weight", None),
                    weights.get(f"{layer_prefix_name}.lora_B.weight", None),
                ),
            )


def inject_adapter_in_model(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    """
    High-level function that injects MixLoraSparseMoe into each block
    of a model.
    """
    model._mixlora_config = config
    # Adjust if your model architecture is different
    for idx, layer in enumerate(model.transformer_blocks):
        _inject_attn_module(idx, layer.attn1, config, weights)
        _inject_mlp_module(idx, layer.ff, config, weights)

#class MixLoraSparseMoe(torch.nn.Module):
#    def __init__(
#        self,
#        base_layer: torch.nn.Module,
#        config: MixLoraConfig,
#    ) -> None:
#        super().__init__()
#
#        self.dtype_: torch.dtype = config.dtype_
#        self.gate_: torch.Tensor = None
#
#        self.base_layer_: torch.nn.Module = base_layer
#        #self.experts_ = nn.ModuleDict()
#        self.experts_: Dict[str, LoraLinear] = {}
#        self.act_fn_ = (
#            ACT2FN[config.act_fn_] if isinstance(config.act_fn_, str) else config.act_fn_
#        )
#        self.num_experts_: int = config.num_experts_
#        self.topk_: int = config.top_k_
#        self.jitter_noise_: float = config.jitter_noise_
#        # if config.model_type_ not in _compatible_model_types:
#        #    raise NotImplementedError()
#        # self.forward_fn_ = getattr(self, _compatible_model_types[config.model_type_])
#
#    def set_gate(self, in_features: int, router_init_range: float = 0.02):
#        self.gate_ = torch.nn.Parameter(
#            torch.empty(
#                self.num_experts_, in_features, dtype=self.dtype_
#            )  # .uniform_(-0.02, 0.02)
#        )
#        torch.nn.init.normal_(
#            self.gate_,
#            mean=0.0,
#            std=router_init_range,
#        )
#
#    def forward_cog(
#        self,
#        expert_mask: torch.Tensor,
#        hidden_states: torch.Tensor,
#        input_dtype: torch.dtype,
#    ) -> torch.Tensor:
#        """
#        Example forward for a CogVideoX feed-forward. Typically:
#          ff.net = nn.Sequential(
#              activation(nn.Linear(in_dim, hidden_dim)),  # net[0]
#              dropout,                        # net[1]
#              nn.Linear(hidden_dim, in_dim),  # net[2]
#              dropout,                        # net[3]
#          )
#
#        We'll call net[0] the "fc1" sub-layer and net[3] the "fc2" sub-layer
#        to be consistent with your other examples.
#        """
#        # The "common_fc1" is the normal forward pass of net[0]
#        # for *all tokens*, ignoring MoE for a moment. Then we add LoRA on top.
#        #   => you can also do the entire net[0..3] for “common” if you prefer,
#        #      but typically we do gating after the first linear.
#
#        # (A) Do the first linear + activation for all tokens:
#        common_fc1 = self.base_layer_.net[0](hidden_states.to(input_dtype))
#        common_act = self.act_fn_(common_fc1)  # net[1] is activation
#
#        # We'll produce final hidden states for each expert separately, then combine.
#        final_expert_states = []
#
#        # (B) For each expert, gather tokens that route to that expert
#        for expert_idx in range(self.num_experts_):
#            # Where does expert_idx appear in "expert_mask"?
#            # expert_mask shape: [num_experts, (batch*seq)]
#            # we get all positions that map to the current expert
#            _, top_positions = torch.where(expert_mask[expert_idx])
#
#            # Grab the LoraLinear modules that correspond to net[0] and net[3], if any:
#            lora_fc1 = self.experts_.get(f"experts.{expert_idx}.net.0", None)
#            lora_fc2 = self.experts_.get(f"experts.{expert_idx}.net.2", None)
#
#            # (B.1) If we have a LoRA for net[0], apply it:
#            if lora_fc1 is not None:
#                # slice out the subset of tokens that go to expert_idx
#                sub_input = common_fc1[top_positions].to(input_dtype)
#                # raw hidden_states (prior to fc1) also needed for LoRA?
#                # depends if your LoraLinear needs the base input
#                # In your code, you do something like:
#                #  lora_fc1.lora_forward(common_fc1_sub, hidden_states_sub)
#                # so we do:
#                sub_original = hidden_states[top_positions].to(input_dtype)
#                fc1_expert = lora_fc1.lora_forward(sub_input, sub_original)
#                # now do the activation
#                fc1_expert = self.act_fn_(fc1_expert)
#            else:
#                # no LoRA, just use the common activation
#                fc1_expert = common_act[top_positions].to(input_dtype)
#
#            # (B.2) Then proceed to net[3] (the second linear).
#            # Actually run the base linear:
#            fc2_base = self.base_layer_.net[2](fc1_expert)
#
#            if lora_fc2 is not None:
#                # If we have LoRA on net[3], apply it
#                # lora_fc2 also needs the “input” to net[3], i.e. fc1_expert
#                fc2_expert = lora_fc2.lora_forward(fc2_base, fc1_expert)
#            else:
#                fc2_expert = fc2_base
#
#            final_expert_states.append(fc2_expert)
#
#        return final_expert_states
#
#    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#        batch_size, sequence_length, hidden_dim = hidden_states.shape
#
#        if self.jitter_noise_ > 0:
#            # Multiply the token inputs by the uniform distribution - adding some noise
#            hidden_states *= torch.empty_like(hidden_states).uniform_(
#                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_
#            )
#
#        input_dtype = hidden_states.dtype
#        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
#        # router_logits: (batch * sequence_length, n_experts)
#        router_logits = F.linear(hidden_states, self.gate_.to(hidden_states))
#
#        routing_weights = F.softmax(router_logits, dim=1, dtype=self.dtype_)
#        routing_weights, selected_experts = torch.topk(routing_weights, self.topk_, dim=-1)
#
#        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#
#        final_hidden_states = torch.zeros(
#            (batch_size * sequence_length, hidden_dim),
#            dtype=self.dtype_,
#            device=hidden_states.device,
#        )
#
#        # One hot encode the selected experts to create an expert mask
#        # this will be used to easily index which expert is going to be sollicitated
#        expert_mask = torch.nn.functional.one_hot(
#            selected_experts, num_classes=self.num_experts_
#        ).permute(2, 1, 0)
#
#        # Perform the computation on each expert
#        # expert_states = self.forward_fn_(
#        expert_states = self.forward_cog(
#            expert_mask,
#            hidden_states,
#            input_dtype,
#        )
#
#        # Unpack
#        for expert_idx in range(self.num_experts_):
#            idx, top_x = torch.where(expert_mask[expert_idx])
#
#            # Index the correct hidden states and compute the expert hidden state for
#            # the current expert. We need to make sure to multiply the output hidden
#            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
#            current_hidden_states = expert_states[expert_idx] * routing_weights[top_x, idx, None]
#
#            # However `index_add_` only support torch tensors for indexing so we'll use
#            # the `top_x` tensor here.
#            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(self.dtype_))
#
#        final_hidden_states = final_hidden_states.reshape(
#            batch_size, sequence_length, hidden_dim
#        ).to(input_dtype)
#
#        return final_hidden_states, routing_weights
#
#
#def _inject_attn_module(
#    layer_idx: int,
#    self_attn: torch.nn.Module,
#    config: MixLoraConfig,
#    weights: Dict[str, torch.Tensor],
#):
#    for proj_name, inject in config.target_modules_.items():
#        if not inject or not hasattr(self_attn, proj_name):
#            continue
#        base_layer = getattr(self_attn, proj_name)
#        layer_prefix_name = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
#        setattr(
#            self_attn,
#            proj_name,
#            LoraLinear(
#                base_layer,
#                config,
#                (
#                    weights[f"{layer_prefix_name}.lora_A.weight"],
#                    weights[f"{layer_prefix_name}.lora_B.weight"],
#                ),
#                # device=base_layer.device
#            ),
#        )
#
#
#def _inject_mlp_module(
#    layer_idx: int,
#    mlp: torch.nn.Module,
#    config: MixLoraConfig,
#    weights: Optional[Dict[str, torch.Tensor]],
#):
#    moe_layer = MixLoraSparseMoe(mlp, config)
#    if weights is None:
#        moe_layer.set_gate(in_features=1920)
#    else:
#        moe_layer.gate_ = weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"].to(
#            config.dtype_
#        )
#
#    if not hasattr(mlp, "mixlora_moes"):
#        mlp.mixlora_moes = {}
#
#    mlp.mixlora_moes[config.adapter_name_] = moe_layer
#    mlp.forward = moe_layer.forward
#
#    base_layer_1 = mlp.net[0].proj
#    base_layer_2 = mlp.net[2]
#
#    lora_ranks = config.lora_r_
#    if isinstance(lora_ranks, int):
#        lora_ranks = [lora_ranks] * config.num_experts_
#    assert (
#        len(lora_ranks) == config.num_experts_
#    ), f"Expected {config.num_experts_} ranks, got {len(lora_ranks)}"
#
#    for expert_idx in range(config.num_experts_):
#
#        layer_1_prefix_name = f"mixlora.layers.{layer_idx}.mlp.net.0.experts.{expert_idx}"
#        layer_2_prefix_name = f"mixlora.layers.{layer_idx}.mlp.net.2.experts.{expert_idx}"
#        #moe_layer.experts_[f"experts.{expert_idx}.net.0"] = LoraLinear(
#        moe_layer.add_module(experts_[f"experts.{expert_idx}.net.0"]) = LoraLinear(
#            base_layer_1,
#            config,
#            lora_ranks[expert_idx],
#            (
#                weights[f"{layer_1_prefix_name}.lora_A.weight"] if weights is not None else None,
#                weights[f"{layer_1_prefix_name}.lora_B.weight"] if weights is not None else None,
#            ),
#            # device=base_layer_1.device
#        )
#        #moe_layer.experts_[f"experts.{expert_idx}.net.2"] = LoraLinear(
#        moe_layer.experts_[f"experts.{expert_idx}.net.2"] = LoraLinear(
#            base_layer_2,
#            config,
#            lora_ranks[expert_idx],
#            (
#                weights[f"{layer_2_prefix_name}.lora_A.weight"] if weights is not None else None,
#                weights[f"{layer_2_prefix_name}.lora_B.weight"] if weights is not None else None,
#            ),
#            # device=base_layer_2.device
#        )
#
#    targets = config.target_modules_
#    if "net" in targets:
#        del targets["net"]
#    for proj_name, inject in targets.items():
#        if not inject or not hasattr(mlp, proj_name):
#            continue
#        base_layer = getattr(mlp, proj_name)
#        for expert_idx in range(config.num_experts_):
#            layer_prefix_name = f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
#            moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = LoraLinear(
#                base_layer,
#                config,
#                (
#                    weights[f"{layer_prefix_name}.lora_A.weight"] if weights is not None else None,
#                    weights[f"{layer_prefix_name}.lora_B.weight"] if weights is not None else None,
#                ),
#            )
#    #        )
#
#
#def inject_adapter_in_model(
#    model: PreTrainedModel,
#    config: MixLoraConfig,
#    weights: Dict[str, torch.Tensor],
#):
#    # config.model_type_ = model.config.model_type
#    model._mixlora_config = config
#    # for idx, layer in enumerate(model.model.layers):
#    for idx, layer in enumerate(model.transformer_blocks):
#        _inject_attn_module(idx, layer.attn1, config, weights)
#        _inject_mlp_module(idx, layer.ff, config, weights)
#        # _inject_attn_module(idx, layer.self_attn, config, weights)
#        # _inject_mlp_module(ix, layer.mlp, config, weights)


def load_adapter_weights(
    name_or_path: str,
    adapter_name: str = "default",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
):
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")

    if device is None:
        device = infer_device()

    with open(name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8") as fp:
        config = MixLoraConfig.from_config(json.load(fp))
        config.adapter_name_ = adapter_name
        config.dtype_ = dtype

    config.check()

    weights: Dict[str, torch.Tensor] = torch.load(
        name_or_path + os.sep + "adapter_model.bin",
        map_location=device,
        weights_only=True,
    )

    return config, weights


def save_adapter_weights(
    model: nn.Module, adapter_name: str = "default", save_directory: str = "./saved_adapter"
):
    """
    Save MixLoRA adapter weights and configuration from `model` into `save_directory`.

    Arguments:
        model: The model that has already been injected with a MixLoRA adapter.
        adapter_name: The adapter name used when injecting LoRA (defaults to "default").
        save_directory: Destination folder for saving the adapter weights and config.
    """
    # 1) Retrieve the adapter config from the model
    if not hasattr(model, "_mixlora_config"):
        raise ValueError(
            "No `_mixlora_config` found on the model. Is this a MixLoRA-injected model?"
        )
    config = model._mixlora_config
    if config.adapter_name_ != adapter_name:
        # Not a hard error in principle; you could store with a different name.
        # But let's enforce consistency for clarity:
        raise ValueError(
            f"Model's config.adapter_name_ is '{config.adapter_name_}' but got '{adapter_name}'."
        )

    # 2) Prepare a dict to store the LoRA weights
    #    We'll mirror the naming scheme used by `_inject_attn_module` and `_inject_mlp_module`.
    lora_state_dict = {}

    # Make sure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Helper function to store the lora_A/lora_B weights
    def _store_lora_weights(prefix: str, lora_linear: LoraLinear):
        lora_state_dict[f"{prefix}.lora_A.weight"] = lora_linear.lora_A.weight.detach().cpu()
        lora_state_dict[f"{prefix}.lora_B.weight"] = lora_linear.lora_B.weight.detach().cpu()

    # 3) Traverse your transformer blocks just like `_inject_*` does
    #    Adjust the iteration logic below to match your actual architecture.
    if not hasattr(model, "transformer_blocks"):
        raise ValueError("Expected `model.transformer_blocks` to exist. Adjust accordingly.")

    for layer_idx, layer in enumerate(model.transformer_blocks):
        # a) For attention submodules
        if hasattr(layer, "attn1"):  # or self_attn, etc. adapt to your naming
            attn = layer.attn1
            # Loop over each possible target module name (e.g. ["q_proj", "k_proj", ...])
            for proj_name, inject in config.target_modules_.items():
                if not inject or not hasattr(attn, proj_name):
                    continue

                maybe_lora_linear = getattr(attn, proj_name)
                if isinstance(maybe_lora_linear, LoraLinear):
                    # Build the same prefix as `_inject_attn_module` does
                    prefix_name = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
                    _store_lora_weights(prefix_name, maybe_lora_linear)

        # b) For MLP submodules
        #    The code in `_inject_mlp_module` sets up a `MixLoraSparseMoe` in `mlp.mixlora_moes[adapter_name]`.
        if hasattr(layer, "ff"):
            mlp = layer.ff
            if hasattr(mlp, "mixlora_moes") and (adapter_name in mlp.mixlora_moes):
                moe_layer = mlp.mixlora_moes[adapter_name]

                # Save the router gate (moe_layer.gate_) if it exists
                if moe_layer.gate_ is not None:
                    lora_state_dict[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = (
                        moe_layer.gate_.detach().cpu()
                    )

                # For each expert in the MixLoraSparseMoe
                for expert_idx in range(moe_layer.num_experts_):
                    # In `_inject_mlp_module`, we stored LoraLinear modules as:
                    #   experts_[f"experts.{expert_idx}.net.0"]
                    #   experts_[f"experts.{expert_idx}.net.2"]
                    #   ... or anything else your model might do

                    # net.0
                    prefix_1 = f"mixlora.layers.{layer_idx}.mlp.net.0.experts.{expert_idx}"
                    if f"experts.{expert_idx}.net.0" in moe_layer.experts_:
                        lora_linear_0 = moe_layer.experts_[f"experts.{expert_idx}.net.0"]
                        if isinstance(lora_linear_0, LoraLinear):
                            _store_lora_weights(prefix_1, lora_linear_0)

                    # net.2
                    prefix_2 = f"mixlora.layers.{layer_idx}.mlp.net.2.experts.{expert_idx}"
                    if f"experts.{expert_idx}.net.2" in moe_layer.experts_:
                        lora_linear_2 = moe_layer.experts_[f"experts.{expert_idx}.net.2"]
                        if isinstance(lora_linear_2, LoraLinear):
                            _store_lora_weights(prefix_2, lora_linear_2)

                    # If your code sets up any additional experts or submodules,
                    # replicate the pattern above for them.

    # 4) Save the config and the adapter state dict
    adapter_config_path = os.path.join(save_directory, "adapter_config.json")
    adapter_model_path = os.path.join(save_directory, "adapter_model.bin")

    # Export the config as JSON
    with open(adapter_config_path, "w", encoding="utf-8") as f:
        json.dump(config.export(), f, indent=2, ensure_ascii=False)

    # Finally, save the LoRA state dict
    torch.save(lora_state_dict, adapter_model_path)

    print(f"Adapter weights saved to: {adapter_model_path}")
    print(f"Adapter config saved to: {adapter_config_path}")


# def save_adapter_weights(
#    name_or_path: str,
#    config: MixLoraConfig,
#    weights: Dict[str, torch.Tensor],
# ):
#    os.makedirs(name_or_path, exist_ok=True)
#
#    with open(name_or_path + os.sep + "adapter_config.json", "w", encoding="utf8") as fp:
#        json.dump(config.export(), fp)
#
#    torch.save(weights, name_or_path + os.sep + "adapter_model.bin")


_compatible_task_types = ["CAUSAL_LM", "QUESTION_ANS"]


@dataclass
class MixLoraModelForCausalLM:
    @staticmethod
    def from_pretrained(
        name_or_path: str,
        *model_args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, MixLoraConfig]:
        config, weights = load_adapter_weights(
            name_or_path,
            adapter_name=kwargs.pop("adapter_name", "default"),
            dtype=kwargs.get("torch_dtype", torch.float32),
        )

        assert config.task_type_ in _compatible_task_types

        model = AutoModelForCausalLM.from_pretrained(config.base_model_, *model_args, **kwargs)

        inject_adapter_in_model(model, config, weights)

        return model, config


mem_config = MixLoraConfig(
    base_model_="THUDM/CogVideoX-2b",
    task_type_="CAUSAL_LM",
    peft_type_="MIXLORA",
    adapter_name_="default",
    model_type_="phi3",
    dtype_=torch.bfloat16,
    use_dora_=False,
    use_rslora_=False,
    lora_init_="original",
    # lora_r_=16,
    lora_r_=[8, 16, 32, 32],
    lora_alpha_=1,
    lora_dropout_=0.1,
    target_modules_={"net": True},
    expert_config_=None,
    router_aux_loss_coef_=0.001,
    router_init_range_=0.02,
    routing_strategy_="mixlora",
    jitter_noise_=0.0,
    router_loss_=True,
    act_fn_="gelu",
    num_experts_=4,
    top_k_=2,
)
if __name__ == "__main__":
    from models.transformer import CogVideoXTransformer3DActionModel, config_2b_iv

    with torch.no_grad(), torch.cuda.amp.autocast():
        model = CogVideoXTransformer3DActionModel(**config_2b_iv).cuda().to(dtype=torch.bfloat16)

        mem_config = MixLoraConfig(
            base_model_="THUDM/CogVideoX-2b",
            task_type_="CAUSAL_LM",
            peft_type_="MIXLORA",
            adapter_name_="default",
            model_type_="phi3",
            dtype_=torch.bfloat16,
            use_dora_=False,
            use_rslora_=False,
            lora_init_="original",
            # lora_r_=16,
            lora_r_=[16, 32, 64, 16],
            lora_alpha_=1,
            lora_dropout_=0.1,
            target_modules_={"net": True},
            expert_config_=None,
            router_aux_loss_coef_=0.001,
            router_init_range_=0.02,
            routing_strategy_="mixlora",
            jitter_noise_=0.0,
            router_loss_=True,
            act_fn_="gelu",
            num_experts_=4,
            top_k_=2,
        )
       # config, weights = load_adapter_weights(
       #     "/home/ss24m050/Documents/CogVideo/my_moe_lora_adapter",
       #     adapter_name="default",
       #     device="cuda",
       #     dtype=torch.bfloat16,
       # )
       # inject_adapter_in_model(model, config, weights)

        inject_adapter_in_model(model, mem_config, None)
        # save_adapter_weights(model, adapter_name="default", save_directory="./my_moe_lora_adapter")
#
#        print(model)
        actions = model.action_encoder.get_dummy_input(batch_size=1, num_frames=48)
        actions = {k: v.cuda() for k, v in actions.items()}
        test_input = torch.randn(1, 49, 32, 20, 30).cuda()
        test_encoder_hidden_states =  None # torch.randn(1, 226, 16, 60, 90).cuda()
        test_timestep = torch.tensor([0]).cuda()
        output = model(test_input, test_encoder_hidden_states, test_timestep, actions=actions)
        print(output.sample.shape)
#
#

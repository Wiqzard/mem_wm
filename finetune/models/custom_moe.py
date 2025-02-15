
import math
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
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
from diffusers.models.attention import Attention, FeedForward


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


class LoraLinear(nn.Module):
    def __init__(
        self,
        # base_layer: nn.Module,
        # config: LoraConfig,
        in_dim,
        out_dim,
        lora_r_: int,  # = None,
        lora_alpha_,
        lora_init_,
        lora_dropout_,
        device: str = None,
        dtype=None,
    ):
        super().__init__()
        # out_dim, in_dim = base_layer.weight.shape

        # self.base_layer_ = base_layer
        self.device_ = torch.device(device) if device else None
        self.dtype_ = dtype

        # self.initializer_ = config.lora_init_
        # self.alpha_ = config.lora_alpha_
        # self.r_ = config.lora_r_ if lora_r_ is None else lora_r_
        self.initializer_ = lora_init_
        self.r_ = lora_r_
        self.alpha_ = lora_alpha_
        self.scaling_ = self.alpha_ / self.r_

        self.in_features_ = in_dim
        self.out_features_ = out_dim

        assert lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=lora_dropout_)

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
        # self.reset_parameters(weight)

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

    def lora_forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        result_lora = (
            self.lora_B(self.lora_A(self.dropout_(hidden_states.to(self.dtype_)))) * self.scaling_
        )
        return residual + result_lora.to(residual.dtype)

    def forward(self, base_layer, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = base_layer(hidden_states)
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


class MoeLoraLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        config: MixLoraConfig,
        inner_dim: int = None,
        device: torch.device = None,
        dtype: str = None,
    ) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32 if dtype is None else dtype
        self.gate_ = torch.nn.Linear(
            in_features,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=self.dtype_,
        )
        self.act_fn_ = ACT2FN[config.act_fn_] if isinstance(config.act_fn_, str) else config.act_fn_

        self.num_experts = config.num_experts_
        expert_ranks = config.lora_r_
        assert len(expert_ranks) == self.num_experts

        fc1_out_dim = inner_dim if inner_dim is not None else  in_features * 4
        self.experts_fc1 = nn.ModuleList([])  # nn.ModuleDict()
        self.experts_fc2 = nn.ModuleList([])  # nn.ModuleDict()
        for rank in expert_ranks:
            self.experts_fc1.append(
                LoraLinear(
                    in_dim=in_features,
                    out_dim=fc1_out_dim,
                    lora_r_=rank,
                    lora_alpha_=config.lora_alpha_,
                    lora_init_=config.lora_init_,
                    lora_dropout_=config.lora_dropout_,
                    device=device,
                    dtype=dtype
                )
            )
            self.experts_fc2.append(
                LoraLinear(
                    in_dim=fc1_out_dim,
                    out_dim=in_features,
                    lora_r_=rank,
                    lora_alpha_=config.lora_alpha_,
                    lora_init_=config.lora_init_,
                    lora_dropout_=config.lora_dropout_,
                    device=device,
                    dtype=dtype
                )
            )
        self.topk_: int = config.top_k_
        self.jitter_noise_: float = config.jitter_noise_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None


    def reset_expert_weights(self):
        # Reset the weights for each expert in experts_fc1 and experts_fc2.
        for expert in self.experts_fc1:
            expert.reset_parameters()
        for expert in self.experts_fc2:
            expert.reset_parameters()


    def _profiling(
        self, batch_size: int, sequence_length: int, selected_experts: torch.Tensor
    ) -> None:
        if not self.router_profile_:
            return

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (router_statistic_[idx] / batch_size) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def forward(
        self,
        ffn_layer,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor = None,
        disable: bool = False
        # input_args,
    ) -> Tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype

        if routing_weights is None:
            hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate_(hidden_states)
            routing_weights_s = F.softmax(router_logits, dim=1, dtype=self.dtype_)
            # sample k 
            # routing_weights_s[k]

            routing_weights, selected_experts = torch.topk(routing_weights_s, self.topk_, dim=-1)
            self._profiling(batch_size, sequence_length, selected_experts)
            #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            #full_routing_weights = torch.zeros_like(router_logits, dtype=routing_weights.dtype, device=routing_weights.device)
            #full_routing_weights.scatter_(1, selected_experts, routing_weights)
        else:
                routing_weights = routing_weights.unsqueeze(0).repeat(batch_size * sequence_length, 1)
                routing_weights, selected_experts = torch.topk(routing_weights, self.topk_, dim=-1)
        
        if torch.all(routing_weights == 0) or disable:
            hidden_states = hidden_states.view(batch_size, sequence_length, hidden_dim)
            for module in ffn_layer.net:
                hidden_states = module(hidden_states)
            return hidden_states
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=self.dtype_,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Perform the computation on each expert
        expert_states = self.forward_cog(
            ffn_layer,
            expert_mask,
            hidden_states,
            input_dtype,
        )

        # Unpack
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_hidden_states = expert_states[expert_idx] * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(self.dtype_))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)
        return final_hidden_states, routing_weights #, #full_routing_weights #router_logits


    def forward_cog(
        self,
        ffn_layer,
        expert_mask,
        hidden_states,
        input_dtype,
    ):
        common_fc1 = ffn_layer.net[0].proj(hidden_states.to(input_dtype))

        final_expert_states = []

        for expert_idx, (expert_fc1, expert_fc2) in enumerate(zip(self.experts_fc1, self.experts_fc2)):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
            act_result = self.act_fn_(
                expert_fc1.lora_forward(
                    _slice_tensor(common_fc1, top_x, input_dtype), lora_data
                )
            )
            act_result = self.act_fn_(_slice_tensor(common_fc1, top_x, input_dtype))

            final_expert_states.append(
                expert_fc2.lora_forward(ffn_layer.net[2](act_result), act_result)
            )

        return final_expert_states


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


def inject_adapter_in_model(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor]=None,
    device: str = None,
    dtype: str = None
):
    """
    High-level function that injects MixLoraSparseMoe into each block
    of a model.
    """
    model._mixlora_config = config
    # Adjust if your model architecture is different
    for idx, layer in enumerate(model.transformer_blocks):
        moe_layer = MoeLoraLayer(in_features=layer.dim,
                                inner_dim=layer.ff_inner_dim,
                                    config=config, 
                                    device=device, 
                                    dtype=dtype)


        layer.moe_layer = moe_layer
        layer.ff.forward = partial(moe_layer.forward, layer.ff)


def fix_routing_weights(model: nn.Module, routing_weights: torch.Tensor):
    num_experts = model._mixlora_config.num_experts_
    assert routing_weights.shape[0] == num_experts
    for idx, layer in enumerate(model.transformer_blocks):
        layer.ff.forward = partial(layer.moe_layer.forward, layer.ff, routing_weights=routing_weights)


def set_adapter_trainable(model: nn.Module, train_experts: bool = True, train_router: bool = True, train_network: bool = False):
    """
    Sets which parameters are trainable based on the adapter training configuration.
    
    By default, the base network parameters (i.e. parameters outside of the adapter modules) are frozen 
    (train_network=False). The adapter is assumed to be injected into each transformer block as a module 
    called `moe_layer` that contains:
    
      - A router/gate module (accessible as `moe_layer.gate_`)
      - Two lists of expert modules (accessible as `moe_layer.experts_fc1` and `moe_layer.experts_fc2`)
    
    The function will:
      - Freeze (or unfreeze) all model parameters according to `train_network`.
      - For each transformer block that contains a moe_layer:
          - Set the gate (router) parameters to be trainable if train_router is True, or frozen otherwise.
          - Set all experts (both in experts_fc1 and experts_fc2) to be trainable if train_experts is True, or frozen otherwise.
    
    Args:
        model (nn.Module): The model that contains adapter modules.
        train_experts (bool): If True, adapter experts are trainable; otherwise they are frozen.
        train_router (bool): If True, the adapter router (gate) is trainable; otherwise it is frozen.
        train_network (bool): If True, also unfreeze (train) the base network parameters. By default, only
                              the adapter parameters will be trainable.
    """
    # First, set the overall training mode for the entire model.
    # If the base network is not to be trained, freeze all parameters.
    if not train_network:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Then, override the setting for adapter parameters.
    # Here we assume that each transformer block is stored in model.transformer_blocks.
    if hasattr(model, "transformer_blocks"):
        for layer in model.transformer_blocks:
            # If the transformer block contains a moe_layer adapter, adjust its parameters.
            if hasattr(layer, "moe_layer"):
                moe_layer = layer.moe_layer

                # For the router/gate: set requires_grad based on train_router flag.
                if hasattr(moe_layer, "gate_"):
                    for param in moe_layer.gate_.parameters():
                        param.requires_grad = train_router

                # For the experts: set requires_grad based on train_experts flag.
                # Assuming moe_layer.experts_fc1 and moe_layer.experts_fc2 are iterables (e.g., lists of modules)
                if hasattr(moe_layer, "experts_fc1"):
                    for expert in moe_layer.experts_fc1:
                        for param in expert.parameters():
                            param.requires_grad = train_experts
                if hasattr(moe_layer, "experts_fc2"):
                    for expert in moe_layer.experts_fc2:
                        for param in expert.parameters():
                            param.requires_grad = train_experts


def get_params(model: nn.Module, router: bool = True, experts: bool = True):
    """
    Returns the parameters of the specified model (usually for the optimizer).
    
    This function iterates through all transformer blocks in the model. 
    For each block that has an injected adapter (stored in `moe_layer`), 
    it collects:
      - The router parameters (moe_layer.gate_) if router=True
      - The expert parameters (moe_layer.experts_fc1 and moe_layer.experts_fc2) if experts=True.
    """
    params = []
    
    if hasattr(model, "transformer_blocks"):
        for block in model.transformer_blocks:
            if hasattr(block, "moe_layer"):
                moe_layer = block.moe_layer
                
                # (1) Router parameters
                if router and hasattr(moe_layer, "gate_"):
                    params.extend(moe_layer.gate_.parameters())
                
                # (2) Expert parameters
                if experts:
                    # experts_fc1
                    if hasattr(moe_layer, "experts_fc1"):
                        # If experts_fc1 is itself a module, just extend with its parameters
                        # (works for ModuleList or ModuleDict):
                        params.extend(moe_layer.experts_fc1.parameters())
                    
                    # experts_fc2
                    if hasattr(moe_layer, "experts_fc2"):
                        # Same approach
                        params.extend(moe_layer.experts_fc2.parameters())
    else:
        print("Warning: Model does not have an attribute 'transformer_blocks'. "
              "No adapter parameters were collected.")
    return params


def disable_adapter(model: nn.Module):
    for idx, layer in enumerate(model.transformer_blocks):
        layer.ff.forward = partial(layer.moe_layer.forward, layer.ff, disable=True)

def activate_adapter(model: nn.Module):
    for idx, layer in enumerate(model.transformer_blocks):
        layer.ff.forward = partial(layer.moe_layer.forward, layer.ff, disable=False)

def reset_expert_weights(model: nn.Module):
    for idx, layer in enumerate(model.transformer_blocks):
        layer.moe_layer.reset_expert_weights()


def save_adapter_weights(model: nn.Module, adapter_name: str="default", save_directory: str="./adapter_weights") -> None:
    """
    Iterates over transformer blocks in `model` and saves the adapter (MoE) weights.
    The saved dictionary will have an entry per block containing:
      - gate: the state_dict of moe_layer.gate_
      - experts_fc1: a list of state_dicts for each expert in moe_layer.experts_fc1
      - experts_fc2: a list of state_dicts for each expert in moe_layer.experts_fc2
    The resulting dict is saved via torch.save.
    """
    if not hasattr(model, "_mixlora_config"):
        raise ValueError(
            "No `_mixlora_config` found on the model. Is this a MixLoRA-injected model?"
        )
    config = model._mixlora_config
    #config.adapter_name = adapter_name
    #if config.adapter_name_ != adapter_name:
    #    raise ValueError(
    #        f"Model's config.adapter_name_ is '{config.adapter_name_}' but got '{adapter_name}'."
    #    )

    # Make sure the directory exists
    os.makedirs(save_directory, exist_ok=True)
    adapter_weights: Dict[str, Any] = {}
    # It is assumed that your model has an attribute 'transformer_blocks'
    for idx, layer in enumerate(model.transformer_blocks):
        if hasattr(layer, "moe_layer"):
            moe_layer = layer.moe_layer
            adapter_weights[f"layer_{idx}"] = {
                "gate": moe_layer.gate_.state_dict(),
                "experts_fc1": [expert.state_dict() for expert in moe_layer.experts_fc1],
                "experts_fc2": [expert.state_dict() for expert in moe_layer.experts_fc2],
            }

    # 4) Save the configuration and the adapter state dict to disk
    adapter_config_path = os.path.join(save_directory, "adapter_config.json")
    adapter_model_path = os.path.join(save_directory, "adapter_model.bin")

    # Export the config as JSON (assuming config.export() returns a serializable dict)
    with open(adapter_config_path, "w", encoding="utf-8") as f:
        json.dump(config.export(), f, indent=2, ensure_ascii=False)

    # Finally, save the adapter state dict
    torch.save(adapter_weights, adapter_model_path)
    print(f"Adapter weights saved to: {adapter_model_path}")
    print(f"Adapter config saved to: {adapter_config_path}")


def load_adapter_weights(
    model: nn.Module, load_directory: str = "./adapter_weights", device=None
) -> None:
    """
    Loads the adapter (MoE) configuration and weights from disk and injects them into `model`.
    
    For each transformer block (assumed to be in model.transformer_blocks) that has saved adapter weights,
    a new MoeLoraLayer is created and its parameters are updated:
      - The gate weights are loaded from the saved state dict.
      - The experts' weights (for both experts_fc1 and experts_fc2) are loaded from the corresponding lists.
    
    Args:
        model (nn.Module): The model to load adapter weights into.
        adapter_name (str): The adapter name (default "default").
        load_directory (str): The directory where adapter_config.json and adapter_model.bin are stored.
    """
    # 1) Verify that the model is MixLoRA-injected and has a matching config.
    if not hasattr(model, "_mixlora_config"):
        adapter_config_path = os.path.join(load_directory, "adapter_config.json")
        # 3) Load the adapter configuration from JSON.
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"Adapter config file not found at: {adapter_config_path}")
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = MixLoraConfig.from_config(config_data)
        inject_adapter_in_model(model=model, config=config, device=device)

    config = model._mixlora_config
    #if config.adapter_name_ != adapter_name:
        #raise ValueError(
            #f"Model's config.adapter_name_ is '{config.adapter_name_}' but got '{adapter_name}'."
        #)

    adapter_model_path = os.path.join(load_directory, "adapter_model.bin")
    # 4) Load the adapter state dictionary.
    if not os.path.exists(adapter_model_path):
        raise FileNotFoundError(f"Adapter model file not found at: {adapter_model_path}")
    adapter_weights: Dict[str, Any] = torch.load(adapter_model_path, map_location=device)

    # 5) For each transformer block with saved adapter weights, create and load a new moe_layer.
    if not hasattr(model, "transformer_blocks"):
        raise ValueError("Expected `model.transformer_blocks` to exist. Adjust accordingly.")

    for idx, layer in enumerate(model.transformer_blocks):
        key = f"layer_{idx}"
        if key in adapter_weights:
            state = adapter_weights[key]
            # Create a new moe_layer using the block's dimensions.
            # It is assumed that each block has attributes 'dim' and 'ff_inner_dim'.
            moe_layer = MoeLoraLayer(
                in_features=layer.dim,
                config=config,
                inner_dim=layer.ff_inner_dim,
                device=device,  # Replace with a device if needed (e.g., torch.device("cuda"))
                dtype=None    # Replace with a dtype if needed (e.g., torch.float32)
            )
            # Load the gate's state.
            moe_layer.gate_.load_state_dict(state["gate"])
            # Load experts_fc1 state for each expert.
            for expert, expert_state in zip(moe_layer.experts_fc1, state["experts_fc1"]):
                expert.load_state_dict(expert_state)
            # Load experts_fc2 state for each expert.
            for expert, expert_state in zip(moe_layer.experts_fc2, state["experts_fc2"]):
                expert.load_state_dict(expert_state)
            # Inject the moe_layer into the transformer block.
            layer.moe_layer = moe_layer.to(device)

    print(f"Adapter weights loaded from: {adapter_model_path}")


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
    #lora_r_=[8, 8, 16, 32],
    lora_r_=[32],
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
    #num_experts_=4,
    num_experts_=1,
    #top_k_=2,
    top_k_=1,
)
if __name__ == "__main__":
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
        lora_r_=[8, 8, 16, 32],
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

    moe_layer = MoeLoraLayer(111, config=mem_config, device=None)
    ff = FeedForward(
        111,
        dropout=0,
        activation_fn="gelu",
        final_dropout=0,
        inner_dim=256,
        bias=True,
    )
    hidden_states = torch.randn((1, 226, 111))
    output = moe_layer(hidden_states, ff)
    print(0)

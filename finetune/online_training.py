import os
import random
import sys
import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diffusers
import torch
import torch.nn.functional as F
import transformers
import wandb
from accelerate import Accelerator
from accelerate.accelerator import DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler

# from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchvision import transforms

from finetune.models.transformer import CogVideoXTransformer3DActionModel, config_2b, config_100m, config_50m, config_2b_iv  # , config_5b
from finetune.models.pipeline import CogVideoXImageToVideoPipeline
from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import I2VDatasetWithActions
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    load_actions,
    load_actions_as_tensors,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
    format_action_string,
)
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)

from finetune.utils.action_utils import generate_action_sequence
from finetune.models.custom_moe import (
    MixLoraConfig,
    MoeLoraLayer,
    inject_adapter_in_model,
    mem_config,
    save_adapter_weights,
    load_adapter_weights,
    set_adapter_trainable,
    fix_routing_weights,
    disable_adapter,
    activate_adapter,
    get_params,
    reset_expert_weights,
)

#def export_to_video(frames, filename, fps=8):
#    """
#    frames: a list of PIL.Image objects
#    filename: the output .mp4 filepath
#    fps: frames per second for the resulting video
#    """
#    # Safety check: if there are no frames, do nothing
#    if not frames:
#        return
#    
#    # Convert all frames to NumPy arrays in RGB
#    frames_np = [np.array(frame.convert("RGB")) for frame in frames]
#    
#    # Use imageio to encode the frames as an mp4 video
#    # Note: 'imageio.mimsave' supports .mp4 if you have ffmpeg installed
#    imageio.mimsave(filename, frames_np, fps=fps)

logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    # UNLOAD_LIST: List[str] = None
    UNLOAD_LIST = ["text_encoder"]

    def __init__(self, args: Args) -> None:
        self.k_mem_steps = 1
        self.num_updates = 1
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()

        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def train_meta_rnn(self):
        """
        After you have added n additional lora weights, learn an optimizer to update them

        """
        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        accelerator = self.accelerator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]
            for step, sample in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate(models_to_accumulate):
                    cut_samples = self.split_sample(
                        sample, self.k_mem_steps, step
                    )  # or change collate_fn

                    if self.args.encode_online:
                        with torch.no_grad():
                            encoded_videos = self.encode_video(batch["videos"].squeeze(1))
                            # encoded_videos = [self.encode_video(video) for video in batch["videos"]]
                            # encoded_videos = torch.stack(encoded_videos).squeeze(1)
                            latent = encoded_videos.to(self.accelerator.device)

                    for i in range(len(cut_samples) - 1):
                        self.memorize(cut_samples[i], k=self.args.k, step=step)
                        for _ in range(self.args.num_updates):
                            loss = self.compute_loss(cut_samples[i + 1])
                            loss.backward()

                    self.meta_optimizer.step()
                    self.optimizer.zero_grad()

                should_run_validation = (
                    self.args.do_validation and global_step % self.args.validation_steps == 0
                )

                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                if global_step >= self.args.train_steps:
                    break

    def train_rl(self):
        """
        Train the model using REINFORCE
        inspiration from 'https://arxiv.org/pdf/1511.06297'
        """
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=0,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        if False: #should_run_validation
            #del loss
            free_memory()
            self.validate_memory(global_step)
        for step, sample in enumerate(self.data_loader):
            self.components.vae.to(self.accelerator.device)
            cut_samples = self.split_sample(sample, num_frames=49)  # or change collate_fn
            for batch in cut_samples:
                print("encode")
                with torch.no_grad():
                    encoded_videos = self.encode_video(batch["videos"].squeeze(1))
                    batch["encoded_videos"] = encoded_videos

            # unload_model(self.components.vae)
            free_memory()

            logs = {}
            for i in range(len(cut_samples) - 1):
                with self.accelerator.no_sync(self.components.transformer):
                    self.memorize(cut_samples[i], k=self.k_mem_steps, step=step)

                    traj_rewards = []
                    log_probs = []
                    for _ in range(self.num_updates):  # this can be done batched
                        clip_loss, routing_weights = self.compute_loss(
                            cut_samples[i + 1], return_routing_weights=True, online=False
                        )

                        traj_rewards.append(-clip_loss)

                        log_probs.append(
                            torch.log(torch.stack([v for k, v in routing_weights.items()]))
                        )
                        # log_probs.append(
                        #    [v for k, v in routing_weights.items()]
                        #    #sum([v for k, v in routing_weights.items()])
                        # )  # [B, L, F,  N]

                    # traj_rewards = torch.stack(traj_rewards)[..., None, None]
                    traj_rewards = torch.stack(traj_rewards)[..., None, None, None].detach()
                    # print(log_probs[0].shape)
                    # print(log_probs[0])
                    log_probs = torch.stack(log_probs)

                    loss = -torch.mean(traj_rewards * log_probs)
                    self.accelerator.backward(loss)

                    print(f"meta_loss_{step}_{i}:   {loss.detach().item()}")
                    logs[f"meta_loss_{i}"] = loss.detach().item()

            self.accelerator.log(logs, step=step)
            # accumulate gradients also in outer loop ! important
            if step % self.args.gradient_accumulation_steps == 0:
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()


            reset_expert_weights(self.components.transformer.module)

            should_run_validation = (
                self.args.do_validation and global_step % self.args.validation_steps == 0
            )

            if False: #should_run_validation
                #del loss
                free_memory()
                self.validate_memory(global_step)


    def memorize(self, sample, k, step=None, prefix="", online=True):
        # sample k timesteps
        # calculate the loss for k timesteps
        # backpropagate the loss
        # if batched:
        #   repeat sample k times and sample batch of k timesteps
        # else:
        # for loop over k timesteps
        accelerator = self.accelerator
        logs = {}
        for i in range(k):
            loss = self.compute_loss(sample, online=online)
            loss.backward()
            self.memory_optimizer.step()
            # self.lr_scheduler.step()
            self.memory_optimizer.zero_grad()

            # print(f"mem_loss_{step}_{i}:   {loss.detach().item()}")
            # logs[f"mem_loss_{step}_{i}"] = loss.detach().item()
            print(f"{prefix}_mem_loss_{i}:   {loss.detach().item()}")
            logs[f"{prefix}_mem_loss_{i}"] = loss.detach().item()

        accelerator.log(logs, step=step)

    def train_router_network(self):
        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        accelerator = self.accelerator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]
            for step, sample in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate(models_to_accumulate):
                    cut_samples = self.split_sample(
                        sample, self.args.k, step
                    )  # or change collate_fn

                    for i in range(len(cut_samples) - 1):
                        self.memorize(cut_samples[i], k=self.args.k, step=step)
                        for _ in range(self.args.num_updates):
                            loss = self.compute_loss(cut_samples[i + 1])
                            loss.backward()

                    self.meta_optimizer.step()
                    self.meta_optimizer.zero_grad()

                should_run_validation = (
                    self.args.do_validation and global_step % self.args.validation_steps == 0
                )

                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                if global_step >= self.args.train_steps:
                    break

    def split_sample(self, samples, num_frames):
        """
        Split the sample into chunks of num_frames along the time dimension.

        For videos, we use a sliding window where the i-th chunk starts at
            start = i * (num_frames - 1)
        and spans frames [start, start+num_frames). The same slicing is applied
        to the actions, but actions are sliced to only include the first (num_frames - 1)
        frames of the chunk.

        Additionally, all keys from the input `samples` dict (except "encoded_videos")
        are added to each output sample.

        Args:
            samples (dict): Dictionary with keys:
                - "videos": Tensor of shape (B, C, T, H, W)
                - "actions": dict of Tensors, each with shape (B, T, ...) for keys such as
                            "wasd", "space", "shift", "mouse_1", "mouse_2", "dx", "dy"
                - Additional metadata keys.
            num_frames (int): Number of frames per video chunk.

        Returns:
            list of dict: Each dict represents a sample chunk with keys:
                - "videos": Tensor of shape (B, C, num_frames, H, W)
                - "first_frames": Tensor of shape (B, C, H, W) (the first frame of the chunk)
                - "actions": Dictionary of sliced actions with time dimension num_frames - 1
                - All other entries from `samples` (except "encoded_videos")
        """
        videos = samples["videos"]  # shape: (B, C, T, H, W)
        actions = samples.get("actions", None)

        B, C, T, H, W = videos.shape
        sample_list = []
        i = 0
        # Slide with a stride of (num_frames - 1)
        while True:
            start = i * (num_frames - 1)
            end = start + num_frames
            if end > T:
                break  # Not enough frames left for a full chunk

            # Slice the video chunk and extract the first frame
            video_chunk = videos[:, :, start:end, :, :]  # (B, C, num_frames, H, W)
            first_frames = video_chunk[:, :, 0, :, :]  # (B, C, H, W)

            # Slice the actions for this chunk (drop the action for the last frame)
            actions_chunk = {}
            if actions is not None:
                for key, tensor in actions.items():
                    # Each tensor is assumed to have shape (B, T, ...)
                    # We slice to include only the first (num_frames - 1) frames
                    actions_chunk[key] = tensor[:, start : end - 1, ...]

            # Build the sample dict for this chunk
            sample = {
                "videos": video_chunk,
                "first_frames": first_frames,
                "actions": actions_chunk,
            }
            # Add any additional entries from samples (except "videos", "actions", and "encoded_videos")
            for key, value in samples.items():
                if key not in ["videos", "actions", "encoded_videos"]:
                    sample[key] = value

            sample_list.append(sample)
            i += 1

        return sample_list

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype
        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name in [
                    "transformer",
                ]:
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "custom":
            inject_adapter_in_model(
                self.components.transformer,
                mem_config,
                weights=None,
                device="cuda",
                dtype=torch.bfloat16,
            )
            set_adapter_trainable(
                self.components.transformer, train_experts=True, train_router=True
            )
            self.__prepare_adapter_saving_loading_hooks()

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            # print(self.components.transformer)

            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_mem_optimizers(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")
        cast_training_params([self.components.transformer], dtype=torch.float32)

        set_adapter_trainable(self.components.transformer)
        router_params = get_params(self.components.transformer, router=True, experts=False)
        # print(router_params)
        router_params = [
            {
                "params": router_params,
                "lr": self.args.learning_rate,
            }
        ]

        lora_params = get_params(self.components.transformer, router=False, experts=True)
        lora_params = {"params": lora_params}

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        self.meta_optimizer = get_optimizer(
            params_to_optimize=router_params,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        self.memory_optimizer = torch.optim.SGD(
            lora_params["params"],  # Provide only the parameter list
            lr=0.0001,  # self.args.lora_learning_rate,  # Assuming a separate learning rate for LoRA
            # momentum=0.9,  # self.args.lora_momentum if hasattr(self.args, "lora_momentum") else 0.9,
            # weight_decay=0.0,
            # (
            #    self.args.lora_weight_decay if hasattr(self.args, "lora_weight_decay") else 0.0
            # ),
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        self.state.num_trainable_parameters = sum(p.numel() for p in lora_params["params"]) + sum(
            p.numel() for p in router_params[0]["params"]
        )

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.meta_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.lr_scheduler = lr_scheduler

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        if False:
            trainable_parameters = list(
                filter(lambda p: p.requires_grad, self.components.transformer.parameters())
            )
            action_encoder_parameters = list(
                filter(
                    lambda p: p.requires_grad,
                    self.components.transformer.action_encoder.parameters(),
                )
            )
            other_parameters = [
                p
                for p in trainable_parameters
                if id(p) not in {id(x) for x in action_encoder_parameters}
            ]
            params_to_optimize = [
                {
                    "params": action_encoder_parameters,
                    "lr": self.args.learning_rate * 2,
                },  # Increased LR for action_encoder
                {
                    "params": other_parameters,
                    "lr": self.args.learning_rate,
                },  # Standard LR for others
            ]
            self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)
        else:
            # For LoRA, we only want to train the LoRA weights
            # For SFT, we want to train all the parameters
            trainable_parameters = list(
                filter(lambda p: p.requires_grad, self.components.transformer.parameters())
            )
            transformer_parameters_with_lr = {
                "params": trainable_parameters,
                "lr": self.args.learning_rate,
            }
            params_to_optimize = [transformer_parameters_with_lr]
            self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training_rl(self) -> None:

        # print(self.components.transformer)
        self.components.transformer, self.meta_optimizer, self.data_loader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.components.transformer,
                self.meta_optimizer,
                self.data_loader,
                self.lr_scheduler,
            )
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_training(self) -> None:

        # print(self.components.transformer)
        self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler
            )
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_for_validation(self):
        validation_prompts = load_prompts(self.args.validation_dir / self.args.validation_prompts)

        if self.args.validation_images is not None:
            validation_images = load_images(self.args.validation_dir / self.args.validation_images)
        else:
            validation_images = [None] * len(validation_prompts)

        if self.args.validation_videos is not None:
            validation_videos = load_videos(self.args.validation_dir / self.args.validation_videos)
        else:
            validation_videos = [None] * len(validation_prompts)

        if self.args.model_type == "wm":
            validation_actions = load_actions(
                self.args.validation_dir / self.args.validation_videos
            )
        else:
            validation_actions = [None] * len(validation_prompts)

        self.state.validation_prompts = validation_prompts
        self.state.validation_images = validation_images
        self.state.validation_videos = validation_videos
        self.state.validation_actions = validation_actions

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "gem-ft"
        self.accelerator.init_trackers(tracker_name, config=self.args.model_dump())

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate_memory(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        #torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

#        if self.state.using_deepspeed:
#            # Can't using model_cpu_offload in deepspeed,
#            # so we need to move all components in pipe to device
#            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
#            self.__move_components_to_device(
#                dtype=self.state.weight_dtype,
#                ignore_list=[
#                    "transformer",
#                ],
#            )
#        else:
#            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
#            pipe.enable_model_cpu_offload(device=self.accelerator.device)
#            # Convert all model weights to training dtype
        pipe = pipe.to(dtype=self.state.weight_dtype)
#
        #################################

        all_processes_artifacts = []
        for i in range(min(2, num_validation_samples - 1)):
            # j = random.randint(0, num_validation_samples - 1)
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on all processes but one
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            prompt = self.state.validation_prompts[i]
            image = self.state.validation_images[i]
            video = self.state.validation_videos[i]  # Ground truth video if available
            action = self.state.validation_actions[i]

            if image is not None:
                image = preprocess_image_with_resize(
                    image, self.state.train_height, self.state.train_width
                )
                # Convert image tensor (C, H, W) to PIL image
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video = preprocess_video_with_resize(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width
                )
                # Convert video tensor (F, C, H, W) to list of PIL images
                video = video.round().clamp(0, 255).to(torch.uint8)
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

            if action is not None:
                action = load_actions_as_tensors(action, num_actions=self.state.train_frames - 1)
                #action = load_actions_as_tensors(action, num_actions=48)
                action_string = format_action_string(action)

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )

            
            num_iterations = 5
            eval_data = {"prompt": prompt, "images": image, "video": video, "actions": action}
            image, video, actions = (
                eval_data["images"],
                eval_data["video"],
                eval_data["actions"],
            )
            #actions = generate_action_sequence(num_frames=48 * num_iterations, command="left-right")
            actions = generate_action_sequence(num_frames=48 * num_iterations, command="forward-backward")
            final_video = []
            for i in range(num_iterations):
                action = actions[i * 48 : (i + 1) * 48]
                #print(len(action))
                action = load_actions_as_tensors(action_list=action, num_actions=len(action))
                eval_data["actions"] = action
                #print(action["wasd"].shape)
                pipe = self.initialize_pipeline()
                latents, video_generate = pipe(
                    num_frames=49, #self.state.train_frames,
                    height=self.state.train_height,
                    width=self.state.train_width,
                    image=eval_data["images"],
                    actions=eval_data["actions"],
                    generator=self.state.generator,
                    guidance_scale=1,
                    dtype=self.components.vae.dtype,
                    #num_inference_steps=50,
                    num_inference_steps=20,
                    return_dict=True,
                ) #.frames[0]
                latents = latents[0]
                video_generate = video_generate[0]


                # PIL images to tensor
                eval_data["images"] = transforms.ToTensor()(video_generate[-1]).unsqueeze(0)                  #video_generate[0] # from pil to tensor 
                eval_data["encoded_videos"] = latents.unsqueeze(0).permute(0, 2, 1, 3, 4) #.unsqueeze(0)
                #self.memorize(eval_data, k=10, step=step, prefix="val", online=False)

                final_video.extend(video_generate)
            
            #final_video = torch.stack(final_video)

            validation_artifacts = [("video", final_video), ]
            #validation_artifacts = self.validation_step(
            #    {"prompt": prompt, "image": image, "video": video, "actions": action}, pipe
            #)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            prompt_filename = string_to_filename(prompt)[:25]
            # Calculate hash of reversed prompt as a unique identifier
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            # Here we keep the original image for reference,
            # and we add an explicit "video_ground_truth" artifact if a GT video is present.
            # -----------------------------------------------------------
            artifacts = {
                "image": {"type": "image", "value": image},
            }
            # ADDED: Store ground-truth video under a more explicit key
            #if video is not None:
            #    artifacts["video_ground_truth"] = {
            #        "type": "video",
            #        "value": video,
            #        "caption": "Ground Truth",
            #    }
            # -----------------------------------------------------------

            # Now include the newly generated artifacts (e.g., predicted video) returned by validation_step
            for idx, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts[f"artifact_{idx}"] = {"type": artifact_type, "value": artifact_value}

            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            # For each artifact (GT video, predicted video(s), or image), save and convert to W&B artifact
            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                caption = value.get("caption", None)

                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.{extension}"
                validation_path = self.args.output_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = str(validation_path / filename)

                if artifact_type == "image":
                    logger.debug(f"Saving image to {filename}")
                    artifact_value.save(filename)
                    artifact_value = wandb.Image(filename)
                elif artifact_type == "video":
                    logger.debug(f"Saving video to {filename}")
                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                    # Use the caption if provided, otherwise fallback to action_string
                    artifact_value = wandb.Video(filename, caption=caption or action_string)

                # Collect W&B artifact objects for gather
                all_processes_artifacts.append(artifact_value)

        # Gather all artifacts across processes
        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)
                    ]
                    video_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)
                    ]
                    # Log them all, including ground-truth videos
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST
            )
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()
#    def validate(self, step: int) -> None:
#        logger.info("Starting validation")
#
#        accelerator = self.accelerator
#        num_validation_samples = len(self.state.validation_prompts)
#
#        if num_validation_samples == 0:
#            logger.warning("No validation samples found. Skipping validation.")
#            return
#
#        self.components.transformer.eval()
#        torch.set_grad_enabled(False)
#
#        memory_statistics = get_memory_statistics()
#        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")
#
#        #####  Initialize pipeline  #####
#        pipe = self.initialize_pipeline()
#
#        if self.state.using_deepspeed:
#            # Can't using model_cpu_offload in deepspeed,
#            # so we need to move all components in pipe to device
#            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
#            self.__move_components_to_device(
#                dtype=self.state.weight_dtype,
#                ignore_list=[
#                    "transformer",
#                ],
#            )
#        else:
#            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
#            pipe.enable_model_cpu_offload(device=self.accelerator.device)
#            # Convert all model weights to training dtype
#            pipe = pipe.to(dtype=self.state.weight_dtype)
#
#        #################################
#
#        all_processes_artifacts = []
#        for i in range(min(8, num_validation_samples - 1)):
#            # j = random.randint(0, num_validation_samples - 1)
#            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
#                # Skip current validation on all processes but one
#                if i % accelerator.num_processes != accelerator.process_index:
#                    continue
#
#            prompt = self.state.validation_prompts[i]
#            image = self.state.validation_images[i]
#            video = self.state.validation_videos[i]  # Ground truth video if available
#            action = self.state.validation_actions[i]
#
#            if image is not None:
#                image = preprocess_image_with_resize(
#                    image, self.state.train_height, self.state.train_width
#                )
#                # Convert image tensor (C, H, W) to PIL image
#                image = image.to(torch.uint8)
#                image = image.permute(1, 2, 0).cpu().numpy()
#                image = Image.fromarray(image)
#
#            if video is not None:
#                video = preprocess_video_with_resize(
#                    video, self.state.train_frames, self.state.train_height, self.state.train_width
#                )
#                # Convert video tensor (F, C, H, W) to list of PIL images
#                video = video.round().clamp(0, 255).to(torch.uint8)
#                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
#
#            if action is not None:
#                action = load_actions_as_tensors(action, num_actions=self.state.train_frames - 1)
#                action_string = format_action_string(action)
#
#            logger.debug(
#                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
#                main_process_only=False,
#            )
#            validation_artifacts = self.validation_step(
#                {"prompt": prompt, "image": image, "video": video, "actions": action}, pipe
#            )
#
#            if (
#                self.state.using_deepspeed
#                and self.accelerator.deepspeed_plugin.zero_stage == 3
#                and not accelerator.is_main_process
#            ):
#                continue
#
#            prompt_filename = string_to_filename(prompt)[:25]
#            # Calculate hash of reversed prompt as a unique identifier
#            reversed_prompt = prompt[::-1]
#            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]
#
#            # Here we keep the original image for reference,
#            # and we add an explicit "video_ground_truth" artifact if a GT video is present.
#            # -----------------------------------------------------------
#            artifacts = {
#                "image": {"type": "image", "value": image},
#            }
#            # ADDED: Store ground-truth video under a more explicit key
#            if video is not None:
#                artifacts["video_ground_truth"] = {
#                    "type": "video",
#                    "value": video,
#                    "caption": "Ground Truth",
#                }
#            # -----------------------------------------------------------
#
#            # Now include the newly generated artifacts (e.g., predicted video) returned by validation_step
#            for idx, (artifact_type, artifact_value) in enumerate(validation_artifacts):
#                artifacts[f"artifact_{idx}"] = {"type": artifact_type, "value": artifact_value}
#
#            logger.debug(
#                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
#                main_process_only=False,
#            )
#
#            # For each artifact (GT video, predicted video(s), or image), save and convert to W&B artifact
#            for key, value in list(artifacts.items()):
#                artifact_type = value["type"]
#                artifact_value = value["value"]
#                caption = value.get("caption", None)
#
#                if artifact_type not in ["image", "video"] or artifact_value is None:
#                    continue
#
#                extension = "png" if artifact_type == "image" else "mp4"
#                filename = f"validation-{step}-{accelerator.process_index}-{prompt_filename}-{hash_suffix}.{extension}"
#                validation_path = self.args.output_dir / "validation_res"
#                validation_path.mkdir(parents=True, exist_ok=True)
#                filename = str(validation_path / filename)
#
#                if artifact_type == "image":
#                    logger.debug(f"Saving image to {filename}")
#                    artifact_value.save(filename)
#                    artifact_value = wandb.Image(filename)
#                elif artifact_type == "video":
#                    logger.debug(f"Saving video to {filename}")
#                    export_to_video(artifact_value, filename, fps=self.args.gen_fps)
#                    # Use the caption if provided, otherwise fallback to action_string
#                    artifact_value = wandb.Video(filename, caption=caption or action_string)
#
#                # Collect W&B artifact objects for gather
#                all_processes_artifacts.append(artifact_value)
#
#        # Gather all artifacts across processes
#        all_artifacts = gather_object(all_processes_artifacts)
#
#        if accelerator.is_main_process:
#            tracker_key = "validation"
#            for tracker in accelerator.trackers:
#                if tracker.name == "wandb":
#                    image_artifacts = [
#                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)
#                    ]
#                    video_artifacts = [
#                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)
#                    ]
#                    # Log them all, including ground-truth videos
#                    tracker.log(
#                        {
#                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
#                        },
#                        step=step,
#                    )
#
#        ##########  Clean up  ##########
#        if self.state.using_deepspeed:
#            del pipe
#            # Unload models except those needed for training
#            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
#        else:
#            pipe.remove_all_hooks()
#            del pipe
#            # Load models except those not needed for training
#            self.__move_components_to_device(
#                dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST
#            )
#            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
#
#            # Change trainable weights back to fp32 to keep with dtype after prepare the model
#            cast_training_params([self.components.transformer], dtype=torch.float32)
#
#        free_memory()
#        accelerator.wait_for_everyone()
#        ################################
#
#        memory_statistics = get_memory_statistics()
#        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
#        torch.cuda.reset_peak_memory_stats(accelerator.device)
#
#        torch.set_grad_enabled(True)
#        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_mem_optimizers()
        self.prepare_for_training_rl()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train_rl()

    # def fit(self):
    #    self.check_setting()
    #    self.prepare_models()
    #    self.prepare_dataset()
    #    self.prepare_trainable_parameters()
    #    self.prepare_optimizer()
    #    self.prepare_for_training()
    #    if self.args.do_validation:
    #        self.prepare_for_validation()
    #    self.prepare_trackers()
    #    self.train()

    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        #components.transformer = CogVideoXTransformer3DActionModel(**config_50m)
        components.transformer = CogVideoXTransformer3DActionModel.from_pretrained(
            self.args.local_path
        )
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    def collate_fn(self, samples: list[dict[str, any]]) -> dict[str, any]:
        ret = {
            "encoded_videos": [],
            "prompt_embedding": [],
            "images": [],
            "actions": {},
            "videos": [],
        }

        for sample in samples:
            # prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            if "video" in sample:
                video = sample["video"]
                ret["videos"].append(video)

            encoded_video = sample["encoded_video"]
            if encoded_video is not None:
                ret["encoded_videos"].append(encoded_video)
            # ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            for key, value in sample["actions"].items():
                if key not in ret["actions"]:
                    ret["actions"][key] = []
                ret["actions"][key].append(value)

        if len(ret["encoded_videos"]) > 0:
            ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["images"] = torch.stack(ret["images"])
        if len(ret["videos"]) > 0:
            ret["videos"] = torch.stack(ret["videos"]).squeeze(1)
        for key in ret["actions"]:
            ret["actions"][key] = torch.cat(ret["actions"][key], dim=0)

        return ret

    def compute_loss(
        self, batch, return_routing_weights: bool = False, online=True
    ) -> torch.Tensor:
        # prompt_embedding = batch["prompt_embedding"]
        if self.args.encode_online and online:
            with torch.no_grad():
                encoded_videos = self.encode_video(batch["videos"].squeeze(1))
                # encoded_videos = [self.encode_video(video) for video in batch["videos"]]
                # encoded_videos = torch.stack(encoded_videos).squeeze(1)
                latent = encoded_videos.to(self.accelerator.device)
        else:
            latent = batch["encoded_videos"].to(self.accelerator.device)

        images = batch["images"].to(self.accelerator.device)
        actions = batch["actions"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        # _, seq_len, _ = prompt_embedding.shape
        # prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # get action embedding
        # uc = bool(torch.rand(1) < 0.2)
        # action_embedding = self.encode_actions(
        #    actions, uc=uc, device=self.accelerator.device, dtype=latent.dtype
        # )

        # prompt_embedding = torch.cat([prompt_embedding, action_embedding], dim=1)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        image_latent_dist = self.components.vae.encode(
            noisy_images.to(dtype=self.components.vae.dtype)
        ).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        ), f"{latent.shape} != {image_latents.shape}"

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )

        model_output = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=None,  # prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
            actions=actions,
            uc=False,  # bool(torch.rand(1) < 0.2)
            mask_ratio=0.0,  # 0.1
        )
        if len(model_output) > 1:
            predicted_noise, routing_weights = model_output
        else:
            predicted_noise = model_output
        # print(routing_weights)

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        if return_routing_weights:
            return loss, routing_weights

        return loss

    

    def validation_step_mem(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        image, video, actions = (
            # eval_data["prompt"],
            eval_data["image"],
            eval_data["video"],
            eval_data["actions"],
        )
        latents, video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            # prompt=prompt,
            image=image,
            actions=actions,
            generator=self.state.generator,
            guidance_scale=1,
            dtype=self.components.vae.dtype,
            output_type="latents&video",
        ).frames[0]

        eval_data["encoded_video"] = latents
        self.memorize(eval_data, k=self.k_mem_steps)

        return [("video", video_generate)]



    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        image, video, actions = (
            # eval_data["prompt"],
            eval_data["image"],
            eval_data["video"],
            eval_data["actions"],
        )

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            # prompt=prompt,
            image=image,
            actions=actions,
            generator=self.state.generator,
            guidance_scale=1,
            dtype=self.components.vae.dtype,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(
                        self.components, name, component.to(self.accelerator.device, dtype=dtype)
                    )

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}"
                        )
            else:
                transformer_ = unwrap_model(
                    self.accelerator, self.components.transformer
                ).__class__.from_pretrained(self.args.model_path, subfolder="transformer")
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(input_dir)
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            or self.accelerator.is_main_process
        ):
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")
        if self.args.model_type == "wm":
            self.dataset = I2VDatasetWithActions(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                # - 1,  # we give action a_{n-1} and generate frame s_n, no need for a_n
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        if self.components.text_encoder is not None:
            self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        if False:
            logger.info("Precomputing latent for video and prompt embedding ...")
            tmp_data_loader = torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_size=1,
                num_workers=0,
                pin_memory=self.args.pin_memory,
            )
            tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
            for _ in tqdm(tmp_data_loader):
                ...
            self.accelerator.wait_for_everyone()
            logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        if self.components.text_encoder is not None:
            unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl",
            timeout=timedelta(seconds=60),
            # backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )

        # mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        # report_to = None if self.args.report_to.lower() == "none" else self.args.report_to
        mixed_precision = self.args.mixed_precision
        report_to = self.args.report_to

        print(f"Using mixed precision: {mixed_precision}")
        print(f"Output directory: {self.args.output_dir}")
        print(f"Logging directory: {logging_dir}")
        print(f"NCCL Timeout: {self.args.nccl_timeout} seconds")

        # Debug: Check distributed environment
        rank = int(os.getenv("RANK", "-1"))
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        print(f"RANK={rank}, WORLD_SIZE={world_size}")
        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )
        print("Distributed initialized.")
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        set_seed(self.args.seed + accelerator.process_index)
        #if self.args.seed is not None:
        #    set_seed(self.args.seed)
        print(f"Using seed: {self.args.seed}")
        print("Distributed initialization complete!")

    def __prepare_adapter_saving_loading_hooks(
        self, adapter_name: str = "default", adapter_directory: str = "./adapter_weights"
    ):
        """
        Registers custom hooks so that adapter weights are saved/loaded in a nice format via the accelerator.

        The save hook calls `save_adapter_weights` (which expects a model with a `transformer_blocks`
        attribute and MixLoRA injection) and writes the adapter configuration and weights to disk.

        The load hook calls `load_adapter_weights` to inject the adapter configuration/weights into the model.
        """

        def save_adapter_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for model in models:
                    # Unwrap the model to check if it is the expected transformer module.
                    unwrapped_model = unwrap_model(self.accelerator, model)
                    expected_type = type(
                        unwrap_model(self.accelerator, self.components.transformer)
                    )
                    if isinstance(unwrapped_model, expected_type):
                        # Save the adapter weights using your provided function.
                        save_adapter_weights(
                            model=unwrapped_model,
                            adapter_name=adapter_name,
                            save_directory=output_dir,
                        )
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")
                    # Pop the weight so that the same model is not saved again.
                    if weights:
                        weights.pop()

        def load_adapter_hook(models, input_dir):
            # When not using DeepSpeed, iterate over the provided models.
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while models:
                    model = models.pop()
                    unwrapped_model = unwrap_model(self.accelerator, model)
                    expected_type = type(
                        unwrap_model(self.accelerator, self.components.transformer)
                    )
                    if isinstance(unwrapped_model, expected_type):
                        load_adapter_weights(
                            model=unwrapped_model,
                            load_directory=input_dir,
                            device=self.args.device,  # or another appropriate device specification
                        )
                    else:
                        raise ValueError(f"Unexpected load model: {model.__class__}")
            else:
                # In DeepSpeed mode, load adapter weights directly into the main transformer.
                unwrapped_transformer = unwrap_model(self.accelerator, self.components.transformer)
                load_adapter_weights(
                    model=unwrapped_transformer,
                    load_directory=input_dir,
                    device=self.args.device,
                )

        # Register the hooks with the accelerator.
        self.accelerator.register_save_state_pre_hook(save_adapter_hook)
        self.accelerator.register_load_state_pre_hook(load_adapter_hook)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

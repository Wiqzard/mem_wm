import random
import sys
import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import I2VDatasetWithResize, T2VDatasetWithResize
from finetune.datasets import I2VDatasetWithActions, WMDataset
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    load_actions,
    load_actions_as_tensors,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
    preprocess_video_with_resize_wm,
    format_action_string
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
import torch.distributed as dist
import numpy as np

def compute_mse(gt_frames, pred_frames):
    """
    Compute mean squared error between two lists of PIL images.

    Args:
        gt_frames (List[PIL.Image.Image]): Ground-truth frames
        pred_frames (List[PIL.Image.Image]): Predicted frames

    Returns:
        float: MSE value (averaged over frames, in the range [0,1]^2)
    """
    if gt_frames is None or pred_frames is None:
        return None

    length = min(len(gt_frames), len(pred_frames))
    if length == 0:
        return None

    total_mse = 0.0
    for i in range(length):
        # Convert PIL Image -> NumPy -> Torch
        gt_tensor = torch.from_numpy(np.array(gt_frames[i], dtype=np.float32))
        pred_tensor = torch.from_numpy(np.array(pred_frames[i], dtype=np.float32))

        # (H, W, C) -> (C, H, W); scale to [0,1]
        gt_tensor = gt_tensor.permute(2, 0, 1) / 255.0
        pred_tensor = pred_tensor.permute(2, 0, 1) / 255.0

        # Per-frame MSE
        frame_mse = torch.mean((gt_tensor - pred_tensor) ** 2).item()
        total_mse += frame_mse

    return total_mse / length

def worker_init_fn(worker_id):
    # Ensure each worker gets a different seed
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = torch.initial_seed() % 2**32
    seed = seed + rank  # Offset seed by process rank
    #seed = torch.initial_seed() % 2**32  # Get base seed
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)



logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
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

        print("init dist")
        self._init_distributed()
        print("init dist done!")
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        #if self.args.seed is not None:
        set_seed(self.args.seed + accelerator.process_index)
        np.random.seed(self.args.seed + accelerator.process_index)
        random.seed(self.args.seed + accelerator.process_index)
        #    set_seed(self.args.seed)

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

        if self.args.model_type == "i2v":
            self.dataset = I2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        elif self.args.model_type == "t2v":
            self.dataset = T2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,
            )
        elif self.args.model_type == "wm":

            self.dataset = WMDataset(
                #**(self.args.model_dump()),
                #device=self.accelerator.device,
                data_root=self.args.data_root,
                video_column=self.args.video_column,
                image_column=self.args.image_column,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                #trainer=self,
            )
            #self.dataset = I2VDatasetWithActions(
            #    **(self.args.model_dump()),
            #    device=self.accelerator.device,
            #    max_num_frames=self.state.train_frames,
            #    # - 1,  # we give action a_{n-1} and generate frame s_n, no need for a_n
            #    height=self.state.train_height,
            #    width=self.state.train_width,
            #    trainer=self,
            #)
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        if self.components.text_encoder is not None:
            self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        if self.components.text_encoder is not None:
            self.components.text_encoder = self.components.text_encoder.to(
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
            worker_init_fn=worker_init_fn
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

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

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        if False:
            trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
            action_encoder_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.action_encoder.parameters()))
            other_parameters = [p for p in trainable_parameters if id(p) not in {id(x) for x in action_encoder_parameters}]
            params_to_optimize = [
                {"params": action_encoder_parameters, "lr": self.args.learning_rate * 2 },  # Increased LR for action_encoder
                {"params": other_parameters, "lr": self.args.learning_rate},  # Standard LR for others
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

    def prepare_for_training(self) -> None:

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

        paired_data = list(zip(validation_prompts, validation_images, validation_videos, validation_actions))
        random.shuffle(paired_data)
        validation_prompts, validation_images, validation_videos, validation_actions = map(list, zip(*paired_data))
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
                if should_run_validation: # and accelerator.is_main_process:
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

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.state.validation_prompts)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Move all necessary components to device if using Deepspeed
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=["transformer", ]
            )
        else:
            # if not using deepspeed, use CPU offload
            pipe.enable_model_cpu_offload(device=self.accelerator.device)
            pipe = pipe.to(dtype=self.state.weight_dtype)
        #################################

        all_processes_artifacts = []

        ### ADDED: We'll accumulate MSE values in a list
        mse_values = []

        for i in range(min(8, num_validation_samples - 1)):
            j = random.randint(0, num_validation_samples - 1)
            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                # Skip current validation on non-selected processes
                if i % accelerator.num_processes != accelerator.process_index:
                    continue

            # Prepare prompt, image, video, action for validation sample
            prompt = self.state.validation_prompts[j]
            image = self.state.validation_images[j]
            video = self.state.validation_videos[j]      # Ground truth video if available
            action = self.state.validation_actions[j]

            if image is not None:
                image = preprocess_image_with_resize(
                    image, self.state.train_height, self.state.train_width
                )
                # Convert image tensor (C, H, W) -> PIL
                image = image.to(torch.uint8)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray(image)

            if video is not None:
                video, image, start_index = preprocess_video_with_resize_wm(
                    video, self.state.train_frames, self.state.train_height, self.state.train_width, random_start=True
                )
                video = video.round().clamp(0, 255).to(torch.uint8)
                # Convert frames -> list of PIL images
                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
                image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

            if action is not None:
                action = load_actions_as_tensors(action, num_actions=self.state.train_frames - 1, start_index=start_index)
                action_string = format_action_string(action)
            else:
                action_string = None

            logger.debug(
                f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                main_process_only=False,
            )

            # Run the validation step (generates predicted video/artifacts)
            validation_artifacts = self.validation_step(
                {"prompt": prompt, "image": image, "video": video, "actions": action}, pipe
            )

            # Deepspeed Zero-3 steps
            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            # Prepare to log the artifacts
            prompt_filename = string_to_filename(prompt)[:25]
            reversed_prompt = prompt[::-1]
            hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

            artifacts = {
                "image": {"type": "image", "value": image},
            }

            # Also store the ground-truth video (if it exists)
            if video is not None:
                artifacts["video_ground_truth"] = {
                    "type": "video",
                    "value": video,
                    "caption": "Ground Truth"
                }

            # The validation step typically returns something like:
            #   [("video", predicted_video_frames), ...]
            # so we store them into `artifacts`
            for idx, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts[f"artifact_{idx}"] = {"type": artifact_type, "value": artifact_value}

            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            # For each artifact, convert to W&B artifacts or video files
            # But before we do that, we can also compute MSE if we have GT + predicted
            gt_frames = artifacts.get("video_ground_truth", {}).get("value", None)
            
            ### ADDED: Find the first predicted "video" artifact to compute MSE
            pred_frames = None
            for k, v in artifacts.items():
                # A typical key might be "artifact_0" with "type"="video"
                # If your `validation_step` returns a single predicted video,
                # you can break right after finding the first.
                if v["type"] == "video" and k != "video_ground_truth":
                    pred_frames = v["value"]
                    break
            
            # Compute MSE if both exist
            if gt_frames is not None and pred_frames is not None:
                sample_mse = compute_mse(gt_frames, pred_frames)
                if sample_mse is not None:
                    mse_values.append(sample_mse)

            # Now proceed with saving the artifacts
            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                caption = value.get("caption", None)

                if artifact_type not in ["image", "video"] or artifact_value is None:
                    continue

                extension = "png" if artifact_type == "image" else "mp4"
                filename = (
                    f"validation-{step}-{accelerator.process_index}-{key}-{hash_suffix}.{extension}"
                )
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

        # Gather all W&B artifacts across processes
        all_artifacts = gather_object(all_processes_artifacts)

        # Gather MSE values across processes
        all_mse_values = gather_object(mse_values)

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

                    # Log images/videos
                    tracker.log(
                        {
                            tracker_key: {
                                "images": image_artifacts,
                                "videos": video_artifacts,
                            },
                        },
                        step=step,
                    )

                    ### ADDED: Log the average MSE
                    if len(all_mse_values) > 0:
                        avg_mse = sum(all_mse_values) / len(all_mse_values)
                        tracker.log({"validation/avg_mse": avg_mse}, step=step)
                        logger.info(f"Validation MSE (step={step}): {avg_mse}")

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Reload needed models
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)
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
#        # if self.components.action_encoder is not None:
#        #     self.components.action_encoder.eval()
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
#                dtype=self.state.weight_dtype, ignore_list=["transformer", ]
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
#        for i in range(min(4, num_validation_samples - 1)):
#            j = random.randint(0, num_validation_samples - 1)
#            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
#                # Skip current validation on all processes but one
#                if i % accelerator.num_processes != accelerator.process_index:
#                    continue
#            #prompt = self.state.validation_prompts[i]
#            #image = self.state.validation_images[i]
#            #video = self.state.validation_videos[i]      # Ground truth video if available
#            #action = self.state.validation_actions[i]
#
#            prompt = self.state.validation_prompts[j]
#            image = self.state.validation_images[j]
#            video = self.state.validation_videos[j]      # Ground truth video if available
#            action = self.state.validation_actions[j]
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
#                video = preprocess_video_with_resize_wm(
#                    video, self.state.train_frames, self.state.train_height, self.state.train_width, random_start=True
#                )[0]
#                # Convert video tensor (F, C, H, W) to list of PIL images
#                video = video.round().clamp(0, 255).to(torch.uint8)
#                # save video to disk
#                #import imageio
#                #import numpy as np
#                #writer = imageio.get_writer(f"video-{j}.mp4", fps=16)
#                #for frame in video:
#                    #frame = frame.permute(1, 2, 0).cpu().numpy()
#                    #writer.append_data(frame)
#                #writer.close()
#
#                video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
#                #video_path = self.args.output_dir / "validation_res" / f"video-{step}-{accelerator.process_index}-{i}.mp4"
#
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
#                    "caption": "Ground Truth"
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
#                filename = (
#                    f"validation-{step}-{accelerator.process_index}-{key}-{hash_suffix}.{extension}"
#                )
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
#            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
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
#
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
#        #if self.components.action_encoder is not None:
#        #    self.components.action_encoder.eval()
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
#                dtype=self.state.weight_dtype, ignore_list=["transformer", ]
#            )
#        else:
#            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
#            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
#            pipe.enable_model_cpu_offload(device=self.accelerator.device)
#
#            # Convert all model weights to training dtype
#            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
#            pipe = pipe.to(dtype=self.state.weight_dtype)
#
#        #################################
#
#        all_processes_artifacts = []
#        for i in range(min(8, num_validation_samples-1)):
#            #j = random.randint(0, num_validation_samples - 1)
#            if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
#                # Skip current validation on all processes but one
#                if i % accelerator.num_processes != accelerator.process_index:
#                    continue
#
#            prompt = self.state.validation_prompts[i]
#            image = self.state.validation_images[i]
#            video = self.state.validation_videos[i]
#            action = self.state.validation_actions[i]
#
#            if image is not None:
#                image = preprocess_image_with_resize(
#                    image, self.state.train_height, self.state.train_width
#                )
#                # Convert image tensor (C, H, W) to PIL images
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
#                # print(action["dx"].shape)
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
#            artifacts = {
#                "image": {"type": "image", "value": image},
#                "video": {"type": "video", "value": video},
#            }
#            for i, (artifact_type, artifact_value) in enumerate(validation_artifacts):
#                artifacts.update(
#                    {f"artifact_{i}": {"type": artifact_type, "value": artifact_value}}
#                )
#            logger.debug(
#                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
#                main_process_only=False,
#            )
#
#            for key, value in list(artifacts.items()):
#                artifact_type = value["type"]
#                artifact_value = value["value"]
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
#                    artifact_value = wandb.Video(filename, caption=action_string)
#
#                all_processes_artifacts.append(artifact_value)
#
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
        self.prepare_optimizer()
        self.prepare_for_training()
        if self.args.do_validation:
            self.prepare_for_validation()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

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

from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register

from finetune.models.transformer import CogVideoXTransformer3DActionModel, config_50m
from finetune.models.pipeline import CogVideoXImageToVideoPipeline


class CogVideoXI2VCustomTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        # components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        # components.text_encoder = T5EncoderModel.from_pretrained(
        #    model_path, subfolder="text_encoder"
        # )

        #components.transformer = CogVideoXTransformer3DActionModel(**config_50m)
        components.transformer = CogVideoXTransformer3DActionModel.from_pretrained(
            self.args.local_path,
            ignore_mismatched_sizes=True 
             # if self.args.local_path is not None else model_path,
            #"/home/ss24m050/Documents/CogVideo/outputs/transformer_2b"
        )  # (**config_5b)

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            # tokenizer=self.components.tokenizer,
            # text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            #action_encoder=self.components.action_encoder,
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        #print(latent.shape)
        #print(latent.mean())
        #print(latent.std())
        #print(latent.min())
        #print(latent.max())
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: list[dict[str, any]]) -> dict[str, any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "actions": {},
               "videos": []}

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
            ret["videos"] = torch.stack(ret["videos"])
        for key in ret["actions"]:
            ret["actions"][key] = torch.cat(ret["actions"][key], dim=0)

        return ret

    def encode_actions(self, actions: Dict[str, Any], uc=False, device=None, dtype=None):
        B, T = actions["dx"].shape
        # self.action_encoder.to(device, dtype=dtype)
        if uc:
            dummy_actions = self.components.action_encoder.get_dummy_input(
                num_frames=T, batch_size=B
            )
            dummy_actions = {k: v.to(device, dtype=dtype) for k, v in dummy_actions.items()}
            encoded_actions = self.components.action_encoder(actions, uc=True)
        else:
            actions = {k: v.to(device, dtype=dtype) for k, v in actions.items()}
            encoded_actions = self.components.action_encoder(actions)
        # self.action_encoder.cpu()
        return encoded_actions

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # prompt_embedding = batch["prompt_embedding"]
        if self.args.encode_online:
            with torch.no_grad():
                encoded_videos = self.encode_video(batch["videos"].squeeze(1))
                #encoded_videos = [self.encode_video(video) for video in batch["videos"]]
                #encoded_videos = torch.stack(encoded_videos).squeeze(1)
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
        )

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


        uc = bool(torch.rand(1) < 0.15)
        encoded_actions = self.components.transformer.encode_actions(
            actions, device=self.accelerator.device, dtype=latent_img_noisy.dtype, uc=uc
        )
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=encoded_actions,  # prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
            actions=actions,
            uc=uc, 
            mask_ratio=0.0, # 0.1
        )[0]

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

        return loss

    @override
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
            guidance_scale=6,
            dtype=self.components.vae.dtype,
            return_dict=True,
            
        )[1][0]
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


register("cogvideox-i2v-wm", "wm", CogVideoXI2VCustomTrainer)
register("cogvideox-i2v-wm", "lora", CogVideoXI2VCustomTrainer)
register("cogvideox-i2v-wm", "sft", CogVideoXI2VCustomTrainer)

register("cogvideox1.5-i2v-wm", "wm", CogVideoXI2VCustomTrainer)
register("cogvideox1.5-i2v-wm", "lora", CogVideoXI2VCustomTrainer)
register("cogvideox1.5-i2v-wm", "sft", CogVideoXI2VCustomTrainer)

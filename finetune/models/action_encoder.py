from typing import Any, Dict, Optional, Tuple, Union
import math

import torch
from torch import nn
import torch.nn.functional as F

from diffusers import ConfigMixin
from einops import rearrange

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
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


def add_sinusoidal_positional_encoding(tensor, dim):
    """
    Adds sinusoidal positional encoding to the given tensor.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (B, T, D),
                               where B is batch size, T is sequence length, and D is feature dim.
        dim (int): The embedding dimension (D).

    Returns:
        torch.Tensor: The tensor with added positional encodings.
    """
    B, T, D = tensor.shape
    assert D == dim, "Tensor embedding dimension must match the specified dim."

    # Compute positional encodings
    position = torch.arange(T, dtype=torch.float32, device=tensor.device).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=tensor.device) * 
                         (-math.log(10000.0) / dim))  # (D/2,)

    pe = torch.zeros(T, dim, device=tensor.device)  # (T, D)
    pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

    # Expand and add positional encoding
    return tensor + pe.unsqueeze(0)  # Shape: (B, T, D)


class ActionEncoder(nn.Module):
    """
    Encodes a dictionary of discrete/continuous actions into a sequence of embeddings.
    """

    def __init__(self, hidden_dim=64, out_dim=None, inner_dim=1920, group_size=4):
        """
        Args:
            hidden_dim (int): Size of the output embedding dimension.
            out_dim (int): Output dimension for final projection.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.out_dim = out_dim

        # Binary action embeddings
        self.w_embedding = nn.Embedding(2, hidden_dim)
        self.a_embedding = nn.Embedding(2, hidden_dim)
        self.s_embedding = nn.Embedding(2, hidden_dim)
        self.d_embedding = nn.Embedding(2, hidden_dim)
        self.space_embedding = nn.Embedding(2, hidden_dim)
        self.shift_embedding = nn.Embedding(2, hidden_dim)
        self.mouse1_embedding = nn.Embedding(2, hidden_dim)
        self.mouse2_embedding = nn.Embedding(2, hidden_dim)

        # MLP for continuous actions (dx, dy)
        def make_mlp(input_dim, hidden_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            )

        self.dx_mlp = make_mlp(1, hidden_dim)
        self.dy_mlp = make_mlp(1, hidden_dim)

        # Temporal embeddings (one learnable embedding per frame index)
        self.frame_embedding_table = nn.Embedding(160, hidden_dim * 10)

        # Mask token for missing actions
        mask_dim = hidden_dim * group_size if group_size is not None else hidden_dim * 10
        self.mask_token = nn.Parameter(torch.randn(1, mask_dim))
        

        #self.discrete_ffn = nn.Linear(4*hidden_dim, out_dim)
        #self.continuous_ffn = nn.Linear(8*hidden_dim, inner_dim)
        # Output transformation (if necessary)
        layer_dim = hidden_dim * 10 if out_dim is None else out_dim
        if out_dim is not None:
            if group_size is None:
                self.final_ffn = nn.Linear(hidden_dim * 10, out_dim)
            else:
                self.final_ffn = nn.Linear(hidden_dim * group_size, out_dim)
                #self.final_ffn = nn.Linear(hidden_dim * group_size, out_dim)
        else:
            self.final_ffn = None

        # Layer normalization
        self.layernorm = T5LayerNorm(layer_dim)

        # Apply weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Custom weight initialization for stable training.
        """
        # Normal initialization for binary action embeddings
        for emb in [
            self.w_embedding, self.a_embedding, self.s_embedding, self.d_embedding,
            self.space_embedding, self.shift_embedding, self.mouse1_embedding, self.mouse2_embedding
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # Kaiming initialization for MLP layers
        for mlp in [self.dx_mlp, self.dy_mlp]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

        # Normal initialization for frame embeddings
        nn.init.normal_(self.frame_embedding_table.weight, mean=0.0, std=0.02)

        # Xavier initialization for final projection (if applicable)
        if self.final_ffn is not None:
            nn.init.xavier_uniform_(self.final_ffn.weight)
            nn.init.zeros_(self.final_ffn.bias)
        
       # if self.continuous_ffn is not None:
       #     nn.init.xavier_uniform_(self.continuous_ffn.weight)
       #     nn.init.zeros_(self.continuous_ffn.bias)
       # 
       # if self.discrete_ffn is not None:
       #     nn.init.xavier_uniform_(self.discrete_ffn.weight)
       #     nn.init.zeros_(self.discrete_ffn.bias)

        # Normal initialization for mask token
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    
    

    def mask_action_sequence(self, action_encoding, mask_ratio=0.15):
        # apply mask to actions (replace encoding with mask token)
        B, T, _ = action_encoding.shape
        mask = torch.bernoulli(torch.full((B, T), mask_ratio)).bool()
        action_encoding[mask] = self.mask_token
        return action_encoding

    def forward(self, actions, uc=False, sequence_length=None, mask_ratio=0.0):
        """
        Args:
            actions is a dictionary containing:
                "w":       (B, T) 
                "a":       (B, T)
                "s":       (B, T)
                "d":       (B, T)
                "e":       (B, T)
                "esc":     (B, T)
                "dwheel":  (B, T)
                "space":   (B, T)
                "shift":   (B, T)
                "mouse_1": (B, T)
                "mouse_2": (B, T)
                "dx":      (B, T)
                "dy":      (B, T)
            uc (bool): If True, return a mask (used in e.g. classifier-free guidance contexts).

        Returns:
            - If group_size is None:  (B, T*10, hidden_dim)
            - If group_size = l:      (B, (T//l)*10, l*hidden_dim)
        """
        B, T = actions["space"].shape

        # -----------------------------------------------------
        # 1) W, A, S, D embeddings (binary: 0 or 1)
        # -----------------------------------------------------
        #wasd = actions["wasd"]  # (B, T, 4)
        #w_emb = self.w_embedding(wasd[:, :, 0].long())  # (B, T, hidden_dim)
        #a_emb = self.a_embedding(wasd[:, :, 1].long())  # (B, T, hidden_dim)
        #s_emb = self.s_embedding(wasd[:, :, 2].long())  # (B, T, hidden_dim)
        #d_emb = self.d_embedding(wasd[:, :, 3].long())  # (B, T, hidden_dim)

        w_emb = self.w_embedding(actions["w"].long())  # (B, T, hidden_dim)
        a_emb = self.a_embedding(actions["a"].long())  # (B, T, hidden_dim)
        s_emb = self.s_embedding(actions["s"].long())  # (B, T, hidden_dim)
        d_emb = self.d_embedding(actions["d"].long())  # (B, T, hidden_dim)

        #e_emb = self.e_embedding(actions["e"].long())  # (B, T, hidden_dim)
        #dwheel_emb = self.dwheel_embedding(actions["dwheel"].long())  # (B, T, hidden_dim)

        # -----------------------------------------------------
        # 2) Space, shift, mouse_1, mouse_2 (binary: 0 or 1)
        # -----------------------------------------------------
        space_emb = self.space_embedding(actions["space"].long())  # (B, T, hidden_dim)
        shift_emb = self.shift_embedding(actions["shift"].long())  # (B, T, hidden_dim)
        mouse1_emb = self.mouse1_embedding(actions["mouse_1"].long())  # (B, T, hidden_dim)
        mouse2_emb = self.mouse2_embedding(actions["mouse_2"].long())  # (B, T, hidden_dim)

        # -----------------------------------------------------
        # 3) dx, dy via MLP (continuous: values in R)
        # -----------------------------------------------------
        dx_in = actions["dx"].unsqueeze(-1)  # (B, T, 1)
        dy_in = actions["dy"].unsqueeze(-1)  # (B, T, 1)

        dx_emb = self.dx_mlp(dx_in.view(B * T, 1)).view(B, T, self.hidden_dim)  # (B, T, hidden_dim)
        dy_emb = self.dy_mlp(dy_in.view(B * T, 1)).view(B, T, self.hidden_dim)  # (B, T, hidden_dim)

        # -----------------------------------------------------
        # 4) Combine: For each frame, produce 1 embedding 
        # -----------------------------------------------------

        all_per_frame = [
            w_emb, a_emb, s_emb, d_emb, space_emb, shift_emb,
            dx_emb, dy_emb, mouse1_emb, mouse2_emb
        ]

        if False:
            all_discrete = torch.cat([w_emb, a_emb, s_emb, d_emb, space_emb, shift_emb, mouse1_emb, mouse2_emb], dim=1)
            out_stacked = add_sinusoidal_positional_encoding(all_discrete, all_discrete.shape[-1]).to(all_discrete.dtype)
            out_seq = rearrange(out_stacked, 'b (t g) c -> b t (g c)', g=self.group_size, t=T*8//self.group_size)
            discrete_seq = self.pad_sequence_with_mask_token(out_seq, sequence_length) # (B, sequence_length, 8*hidden_dim)
            discrete_seq = self.discrete_ffn(discrete_seq)


            all_continuous = torch.cat([dx_emb, dy_emb], dim=2) # (B, T, 2*hidden_dim)
            continuous_seq = rearrange(all_continuous, 'b (t g) c -> b t (g c)', g=self.group_size, t=T//self.group_size)
            continuous_seq = self.continuous_ffn(continuous_seq)

            return discrete_seq, continuous_seq


        if self.group_size is not None:
            assert T % self.group_size == 0, "T must be divisible by group_size"
            #out_stacked = torch.cat(all_per_frame, dim=2) # (B, T, 10 * self.hidden_dim)
            #out_seq = add_sinusoidal_positional_encoding(out_stacked, out_stacked.shape[-1])
            #out_seq = rearrange(out_stacked, 'b t (s c) -> b (t s) c', s=10, c=self.hidden_dim) # (B, T*10, hidden_dim)
            #out_seq = rearrange(out_seq, 'b (t g) c -> b t (g c)', g=4, )
            out_stacked = torch.cat(all_per_frame, dim=1) # (b, t*10, self.hidden_dim)
            out_seq = add_sinusoidal_positional_encoding(out_stacked, out_stacked.shape[-1])
            out_seq = out_seq.to(out_stacked.dtype)
            out_seq = rearrange(out_seq, 'b (t g) c -> b t (g c)', g=self.group_size, t=T*10//self.group_size)
            if sequence_length is not None and out_seq.shape[1] < sequence_length:
                out_seq = self.pad_sequence_with_mask_token(out_seq, sequence_length) # (B, sequence_length, 10*hidden_dim)
            out_seq = self.final_ffn(out_seq)
            return out_seq
            
        out_stacked = torch.cat(all_per_frame, dim=2) # (B, T, 10 * self.hidden_dim)

        # -----------------------------------------------------
        # 5) Mask action sequence
        # -----------------------------------------------------
        if mask_ratio > 0.0:
            out_stacked = self.mask_action_sequence(out_stacked, mask_ratio=mask_ratio)
        
        if uc:
            out_stacked = self.mask_token.repeat(B, T, 1)

        # -----------------------------------------------------
        # 6) Temporal (frame) embedding => (B, T, hidden_dim)
        # -----------------------------------------------------
        frame_ids = torch.arange(T, device=space_emb.device)  # (T,)
        frame_embs = self.frame_embedding_table(frame_ids)  # (T, hidden_dim * 10)
        frame_embs = frame_embs.unsqueeze(0).expand(B, T, self.hidden_dim * 10)
        out_with_time = out_stacked + frame_embs

        # -----------------------------------------------------
        # 6) Pad sequence with mask token
        # -----------------------------------------------------
        if sequence_length is not None:
            out_with_time = self.pad_sequence_with_mask_token(out_with_time, sequence_length) # (B, sequence_length, 10*hidden_dim)

        # -----------------------------------------------------
        # 8) Final FFN and LayerNorm
        # -----------------------------------------------------

        if hasattr(self, "final_ffn"):
            out_with_time = self.final_ffn(out_with_time)


        dim = 10 * self.hidden_dim if self.final_ffn is None else self.final_ffn.out_features

        out_seq = add_sinusoidal_positional_encoding(out_with_time, dim)

        out_seq = self.layernorm(out_seq)

        return out_seq 


    @staticmethod
    def from_pretrained(self,  path:str) -> "ActionEncoder":
        """
        Load the model from a file.
        """
        model = torch.load(path)
        return model 
        
    
    def get_dummy_input(self, batch_size=2, num_frames=20):
        """
        Returns a dummy input for the model.
        """
        # just w
        wasd = torch.zeros(batch_size, num_frames, 4).float()
        wasd[:, :, 1] = 1
        space = torch.zeros(batch_size, num_frames)
        shift = torch.zeros(batch_size, num_frames)
        mouse_1 = torch.zeros(batch_size, num_frames)
        mouse_2 = torch.zeros(batch_size, num_frames).float()
        dx = torch.zeros(batch_size, num_frames)
        dy = torch.zeros(batch_size, num_frames)
        actions = {
            "wasd": wasd,
            "space": space,
            "shift": shift,
            "mouse_1": mouse_1,
            "mouse_2": mouse_2,
            "dx": dx,
            "dy": dy,
        }
        return actions        
    
    def pad_sequence_with_mask_token(self, sequence, sequence_length: int):
        """
        Pad the sequence with mask tokens to the specified length.
        """
        B, T, _ = sequence.shape
        if T < sequence_length:
            mask_token = self.mask_token.repeat(B, sequence_length - T, 1)
            sequence = torch.cat([sequence, mask_token], dim=1)
        return sequence


    #def forward(self, actions, uc=False):
        #"""
        #actions is a dictionary containing:
            #"wasd":    (B, T, 4)
            #"space":   (B, T)
            #"shift":   (B, T)
            #"mouse_1": (B, T)
            #"mouse_2": (B, T)
            #"dx":      (B, T)
            #"dy":      (B, T)

        #Returns:
            #By default: (B, T*7, hidden_dim)
            #If group_size = l is specified:
               #(B, (T//l)*7, l * hidden_dim)
        #"""
        #B, T, _ = actions["wasd"].shape  # wasd has shape (B, T, 4)

        #if uc:
            #return self.mask_token.repeat(B, T // self.group_size * 7, 1)

        ## -----------------------------------------------------
        ## 1) Compute WASD embedding
        ##    For each time step, we have a one-hot of shape (4,).
        ##    We'll pick the embeddings for those pressed, sum them, then average.
        ## -----------------------------------------------------
        #wasd_flat = actions["wasd"].view(B * T, 4)
        ## embed all 4 directions: shape => (4, hidden_dim)
        #all_wasd_embeds = self.wasd_embedding_table(
            #torch.arange(4, device=wasd_flat.device)
        #)  # (4, hidden_dim)
        ## Expand to (B*T, 4, hidden_dim)
        #all_wasd_embeds_expanded = all_wasd_embeds.unsqueeze(0).expand(B * T, 4, self.hidden_dim)
        ## Weighted sum by the one-hot
        #weights = wasd_flat.unsqueeze(-1)  # (B*T, 4, 1)
        #weighted_wasd = all_wasd_embeds_expanded * weights
        #sum_wasd = weighted_wasd.sum(dim=1)  # (B*T, hidden_dim)
        #pressed_counts = wasd_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
        #wasd_emb = sum_wasd / pressed_counts  # (B*T, hidden_dim)
        #wasd_emb = wasd_emb.view(B, T, self.hidden_dim)  # (B, T, hidden_dim)

        ## -----------------------------------------------------
        ## 2) space, shift, mouse_1, mouse_2
        ## -----------------------------------------------------
        #space_emb = self.space_embedding.view(1, 1, self.hidden_dim)
        #shift_emb = self.shift_embedding.view(1, 1, self.hidden_dim)
        #mouse1_emb = self.mouse1_embedding.view(1, 1, self.hidden_dim)
        #mouse2_emb = self.mouse2_embedding.view(1, 1, self.hidden_dim)

        #space_out = actions["space"].unsqueeze(-1) * space_emb  # (B, T, hidden_dim)
        #shift_out = actions["shift"].unsqueeze(-1) * shift_emb  # (B, T, hidden_dim)
        #mouse1_out = actions["mouse_1"].unsqueeze(-1) * mouse1_emb  # (B, T, hidden_dim)
        #mouse2_out = actions["mouse_2"].unsqueeze(-1) * mouse2_emb  # (B, T, hidden_dim)

        ## -----------------------------------------------------
        ## 3) dx, dy via MLP => (B, T, hidden_dim)
        ## -----------------------------------------------------
        #dx_in = actions["dx"].unsqueeze(-1)  # (B, T, 1)
        #dy_in = actions["dy"].unsqueeze(-1)  # (B, T, 1)

        #dx_flat = dx_in.reshape(B * T, 1)
        #dy_flat = dy_in.reshape(B * T, 1)

        #dx_emb = self.dx_mlp(dx_flat).view(B, T, self.hidden_dim)
        #dy_emb = self.dy_mlp(dy_flat).view(B, T, self.hidden_dim)

        ## -----------------------------------------------------
        ## 4) Temporal (frame) embedding => (B, T, hidden_dim)
        ## -----------------------------------------------------
        #frame_ids = torch.arange(T, device=wasd_emb.device)  # (T,)
        #frame_embs = self.frame_embedding_table(frame_ids)  # (T, hidden_dim)
        #frame_embs = frame_embs.unsqueeze(0).expand(B, T, self.hidden_dim)

        ## -----------------------------------------------------
        ## 5) Combine: For each frame, produce 7 embeddings
        ##    Then add frame_embs to each => shape (B, T, 7, hidden_dim)
        ## -----------------------------------------------------
        #all_per_frame = [wasd_emb, space_out, shift_out, dx_emb, dy_emb, mouse1_out, mouse2_out]
        ## (B, T, 7, hidden_dim)
        #out_stacked = torch.stack(all_per_frame, dim=2)
        ## Add the frame embedding to each of the 7 slots
        #out_with_time = out_stacked + frame_embs.unsqueeze(2)

        ## -----------------------------------------------------
        ## Default flatten => (B, T*7, hidden_dim)
        ## -----------------------------------------------------
        #if self.group_size is None:
            ## Just flatten time dimension and the 7 slots
            #out_seq = out_with_time.view(B, T * 7, self.hidden_dim)
        #else:
            ## -------------------------------------------------
            ## 6) Group frames by group_size = l
            ##    We assume T is divisible by l.
            ##    The grouping:
            ##        shape => (B, T, 7, hidden_dim)
            ##        reshape => (B, T//l, l, 7, hidden_dim)
            ##        reorder => (B, T//l, 7, l, hidden_dim)
            ##        flatten => (B, (T//l)*7, l * hidden_dim)
            ## -------------------------------------------------
            #l = self.group_size
            #assert T % l == 0, "T must be divisible by group_size"
            #G = T // l  # number of groups

            #out_with_time = out_with_time.view(B, G, l, 7, self.hidden_dim)
            ## permute to (B, G, 7, l, C)
            #out_with_time = out_with_time.permute(0, 1, 3, 2, 4)
            ## flatten G*7 => (G*7), and flatten l*C => (l*C)
            #out_seq = out_with_time.reshape(
                #B, G * 7, l * self.hidden_dim  # time dimension after grouping  # channel dimension
            #)

        #return out_seq


# -----------------------
# Example usage:
if __name__ == "__main__":
    B, T = 2, 80  # batch size=2, 6 frames
    hidden_dim = 4096//8 
    group_size = 8  # group every 2 frames

    # Create random actions
    wasd = F.one_hot(torch.randint(0, 4, (B, T)), num_classes=4).float()  # shape (B,T,4)
    space = torch.randint(0, 2, (B, T)).float()
    shift = torch.randint(0, 2, (B, T)).float()
    mouse_1 = torch.randint(0, 2, (B, T)).float()
    mouse_2 = torch.randint(0, 2, (B, T)).float()
    dx = torch.randn(B, T)
    dy = torch.randn(B, T)

    actions = {
        "wasd": wasd,
        "space": space,
        "shift": shift,
        "mouse_1": mouse_1,
        "mouse_2": mouse_2,
        "dx": dx,
        "dy": dy,
    }

    # If group_size is None, shape = (B, T*7, hidden_dim)
    # If group_size = l, shape = (B, (T//l)*7, l*hidden_dim)
    model = ActionEncoder(hidden_dim=hidden_dim, num_frames=T, group_size=group_size)
    out_seq = model(actions)
    print("Output shape:", out_seq.shape)
    # If group_size=2 => out_seq.shape = (2, (6/2)*7, 2*8) = (2, 21, 16)


#
#
## -----------------------
## Example usage:
# if __name__ == "__main__":
#    B, T = 2, 81  # batch size=2, 3 frames
#    hidden_dim = 16
#
#    # Create random actions (one-hot for wasd, random 0/1 for others, random floats for dx/dy)
#    wasd = F.one_hot(
#        torch.randint(
#            0,
#            4,
#            (
#                B,
#                T,
#            ),
#        ),
#        num_classes=4,
#    ).float()  # shape (B,T,4)
#    space = torch.randint(0, 2, (B, T)).float()
#    shift = torch.randint(0, 2, (B, T)).float()
#    mouse_1 = torch.randint(0, 2, (B, T)).float()
#    mouse_2 = torch.randint(0, 2, (B, T)).float()
#    dx = torch.randn(B, T)
#    dy = torch.randn(B, T)
#
#    actions = {
#        "wasd": wasd,
#        "space": space,
#        "shift": shift,
#        "mouse_1": mouse_1,
#        "mouse_2": mouse_2,
#        "dx": dx,
#        "dy": dy,
#    }
#
#    model = ActionEncoder(hidden_dim=hidden_dim, num_frames=T)
#    out_seq = model(actions)  # shape => (B, T*7, hidden_dim)
#    print("Output shape:", out_seq.shape)
#    # e.g. => (2, 3*7, 16) = (2, 21, 16)
#
#
#@maybe_allow_in_graph
#class CogVideoXBlock(nn.Module):
#    r"""
#    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.
#
#    Parameters:
#        dim (`int`):
#            The number of channels in the input and output.
#        num_attention_heads (`int`):
#            The number of heads to use for multi-head attention.
#        attention_head_dim (`int`):
#            The number of channels in each head.
#        time_embed_dim (`int`):
#            The number of channels in timestep embedding.
#        dropout (`float`, defaults to `0.0`):
#            The dropout probability to use.
#        activation_fn (`str`, defaults to `"gelu-approximate"`):
#            Activation function to be used in feed-forward.
#        attention_bias (`bool`, defaults to `False`):
#            Whether or not to use bias in attention projection layers.
#        qk_norm (`bool`, defaults to `True`):
#            Whether or not to use normalization after query and key projections in Attention.
#        norm_elementwise_affine (`bool`, defaults to `True`):
#            Whether to use learnable elementwise affine parameters for normalization.
#        norm_eps (`float`, defaults to `1e-5`):
#            Epsilon value for normalization layers.
#        final_dropout (`bool` defaults to `False`):
#            Whether to apply a final dropout after the last feed-forward layer.
#        ff_inner_dim (`int`, *optional*, defaults to `None`):
#            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
#        ff_bias (`bool`, defaults to `True`):
#            Whether or not to use bias in Feed-forward layer.
#        attention_out_bias (`bool`, defaults to `True`):
#            Whether or not to use bias in Attention output projection layer.
#    """
#
#    def __init__(
#        self,
#        dim: int,
#        num_attention_heads: int,
#        attention_head_dim: int,
#        time_embed_dim: int,
#        dropout: float = 0.0,
#        activation_fn: str = "gelu-approximate",
#        attention_bias: bool = False,
#        qk_norm: bool = True,
#        norm_elementwise_affine: bool = True,
#        norm_eps: float = 1e-5,
#        final_dropout: bool = True,
#        ff_inner_dim: Optional[int] = None,
#        ff_bias: bool = True,
#        attention_out_bias: bool = True,
#    ):
#        super().__init__()
#
#        # 1. Self Attention
#        self.norm1 = CogVideoXLayerNormZero(
#            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
#        )
#
#        self.attn1 = Attention(
#            query_dim=dim,
#            dim_head=attention_head_dim,
#            heads=num_attention_heads,
#            qk_norm="layer_norm" if qk_norm else None,
#            eps=1e-6,
#            bias=attention_bias,
#            out_bias=attention_out_bias,
#            processor=CogVideoXAttnProcessor2_0(),
#        )
#
#        # 2. Feed Forward
#        self.norm2 = CogVideoXLayerNormZero(
#            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True
#        )
#
#        self.ff = FeedForward(
#            dim,
#            dropout=dropout,
#            activation_fn=activation_fn,
#            final_dropout=final_dropout,
#            inner_dim=ff_inner_dim,
#            bias=ff_bias,
#        )
#
#    def forward(
#        self,
#        hidden_states: torch.Tensor,
#        encoder_hidden_states: torch.Tensor,
#        temb: torch.Tensor,
#        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#    ) -> torch.Tensor:
#        text_seq_length = encoder_hidden_states.size(1)
#
#        # norm & modulate
#        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
#            hidden_states, encoder_hidden_states, temb
#        )
#
#        # attention
#        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
#            hidden_states=norm_hidden_states,
#            encoder_hidden_states=norm_encoder_hidden_states,
#            image_rotary_emb=image_rotary_emb,
#        )
#
#        hidden_states = hidden_states + gate_msa * attn_hidden_states
#        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
#
#        # norm & modulate
#        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
#            hidden_states, encoder_hidden_states, temb
#        )
#
#        # feed-forward
#        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
#        ff_output = self.ff(norm_hidden_states)
#
#        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
#        encoder_hidden_states = (
#            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
#        )
#
#        return hidden_states, encoder_hidden_states
#

#class CogVideoXTransformer3DModelAction(ModelMixin, ConfigMixin, PeftAdapterMixin):
    #"""
    #A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    #Parameters:
        #num_attention_heads (`int`, defaults to `30`):
            #The number of heads to use for multi-head attention.
        #attention_head_dim (`int`, defaults to `64`):
            #The number of channels in each head.
        #in_channels (`int`, defaults to `16`):
            #The number of channels in the input.
        #out_channels (`int`, *optional*, defaults to `16`):
            #The number of channels in the output.
        #flip_sin_to_cos (`bool`, defaults to `True`):
            #Whether to flip the sin to cos in the time embedding.
        #time_embed_dim (`int`, defaults to `512`):
            #Output dimension of timestep embeddings.
        #ofs_embed_dim (`int`, defaults to `512`):
            #Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        #text_embed_dim (`int`, defaults to `4096`):
            #Input dimension of text embeddings from the text encoder.
        #num_layers (`int`, defaults to `30`):
            #The number of layers of Transformer blocks to use.
        #dropout (`float`, defaults to `0.0`):
            #The dropout probability to use.
        #attention_bias (`bool`, defaults to `True`):
            #Whether to use bias in the attention projection layers.
        #sample_width (`int`, defaults to `90`):
            #The width of the input latents.
        #sample_height (`int`, defaults to `60`):
            #The height of the input latents.
        #sample_frames (`int`, defaults to `49`):
            #The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            #instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            #but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            #K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        #patch_size (`int`, defaults to `2`):
            #The size of the patches to use in the patch embedding layer.
        #temporal_compression_ratio (`int`, defaults to `4`):
            #The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        #max_text_seq_length (`int`, defaults to `226`):
            #The maximum sequence length of the input text embeddings.
        #activation_fn (`str`, defaults to `"gelu-approximate"`):
            #Activation function to use in feed-forward.
        #timestep_activation_fn (`str`, defaults to `"silu"`):
            #Activation function to use when generating the timestep embeddings.
        #norm_elementwise_affine (`bool`, defaults to `True`):
            #Whether to use elementwise affine in normalization layers.
        #norm_eps (`float`, defaults to `1e-5`):
            #The epsilon value to use in normalization layers.
        #spatial_interpolation_scale (`float`, defaults to `1.875`):
            #Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        #temporal_interpolation_scale (`float`, defaults to `1.0`):
            #Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    #"""

    #_supports_gradient_checkpointing = True

    ## @register_to_config
    #def __init__(
        #self,
        #num_attention_heads: int = 30,
        #attention_head_dim: int = 64,
        #in_channels: int = 16,
        #out_channels: Optional[int] = 16,
        #flip_sin_to_cos: bool = True,
        #freq_shift: int = 0,
        #time_embed_dim: int = 512,
        #ofs_embed_dim: Optional[int] = None,
        #text_embed_dim: int = 4096,
        #num_layers: int = 30,
        #dropout: float = 0.0,
        #attention_bias: bool = True,
        #sample_width: int = 90,
        #sample_height: int = 60,
        #sample_frames: int = 49,
        #patch_size: int = 2,
        #patch_size_t: Optional[int] = None,
        #temporal_compression_ratio: int = 4,
        #max_text_seq_length: int = 226,
        #activation_fn: str = "gelu-approximate",
        #timestep_activation_fn: str = "silu",
        #norm_elementwise_affine: bool = True,
        #norm_eps: float = 1e-5,
        #spatial_interpolation_scale: float = 1.875,
        #temporal_interpolation_scale: float = 1.0,
        #use_rotary_positional_embeddings: bool = False,
        #use_learned_positional_embeddings: bool = False,
        #patch_bias: bool = True,
    #):
        #super().__init__()
        #inner_dim = num_attention_heads * attention_head_dim

        #if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            #raise ValueError(
                #"There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                #"embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                #"issue at https://github.com/huggingface/diffusers/issues."
            #)
        #self.action_encoder = ActionEncoder(32, 81)
        ## 1. Patch embedding
        #self.patch_embed = CogVideoXPatchEmbed(
            #patch_size=patch_size,
            #patch_size_t=patch_size_t,
            #in_channels=in_channels,
            #embed_dim=inner_dim,
            #text_embed_dim=text_embed_dim,
            #bias=patch_bias,
            #sample_width=sample_width,
            #sample_height=sample_height,
            #sample_frames=sample_frames,
            #temporal_compression_ratio=temporal_compression_ratio,
            #max_text_seq_length=max_text_seq_length,
            #spatial_interpolation_scale=spatial_interpolation_scale,
            #temporal_interpolation_scale=temporal_interpolation_scale,
            #use_positional_embeddings=not use_rotary_positional_embeddings,
            #use_learned_positional_embeddings=use_learned_positional_embeddings,
        #)
        #self.embedding_dropout = nn.Dropout(dropout)

        ## 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        #self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        #self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        #self.ofs_proj = None
        #self.ofs_embedding = None
        #if ofs_embed_dim:
            #self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            #self.ofs_embedding = TimestepEmbedding(
                #ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            #)  # same as time embeddings, for ofs

        ## 3. Define spatio-temporal transformers blocks
        #self.transformer_blocks = nn.ModuleList(
            #[
                #CogVideoXBlock(
                    #dim=inner_dim,
                    #num_attention_heads=num_attention_heads,
                    #attention_head_dim=attention_head_dim,
                    #time_embed_dim=time_embed_dim,
                    #dropout=dropout,
                    #activation_fn=activation_fn,
                    #attention_bias=attention_bias,
                    #norm_elementwise_affine=norm_elementwise_affine,
                    #norm_eps=norm_eps,
                #)
                #for _ in range(num_layers)
            #]
        #)
        #self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        ## 4. Output blocks
        #self.norm_out = AdaLayerNorm(
            #embedding_dim=time_embed_dim,
            #output_dim=2 * inner_dim,
            #norm_elementwise_affine=norm_elementwise_affine,
            #norm_eps=norm_eps,
            #chunk_dim=1,
        #)

        #if patch_size_t is None:
            ## For CogVideox 1.0
            #output_dim = patch_size * patch_size * out_channels
        #else:
            ## For CogVideoX 1.5
            #output_dim = patch_size * patch_size * patch_size_t * out_channels

        #self.proj_out = nn.Linear(inner_dim, output_dim)

        #self.gradient_checkpointing = False

    #def _set_gradient_checkpointing(self, module, value=False):
        #self.gradient_checkpointing = value

    #@property
    ## Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    #def attn_processors(self) -> Dict[str, AttentionProcessor]:
        #r"""
        #Returns:
            #`dict` of attention processors: A dictionary containing all attention processors used in the model with
            #indexed by its weight name.
        #"""
        ## set recursively
        #processors = {}

        #def fn_recursive_add_processors(
            #name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]
        #):
            #if hasattr(module, "get_processor"):
                #processors[f"{name}.processor"] = module.get_processor()

            #for sub_name, child in module.named_children():
                #fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            #return processors

        #for name, module in self.named_children():
            #fn_recursive_add_processors(name, module, processors)

        #return processors

    ## Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    #def set_attn_processor(
        #self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    #):
        #r"""
        #Sets the attention processor to use to compute attention.

        #Parameters:
            #processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                #The instantiated processor class or a dictionary of processor classes that will be set as the processor
                #for **all** `Attention` layers.

                #If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                #processor. This is strongly recommended when setting trainable attention processors.

        #"""
        #count = len(self.attn_processors.keys())

        #if isinstance(processor, dict) and len(processor) != count:
            #raise ValueError(
                #f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                #f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            #)

        #def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            #if hasattr(module, "set_processor"):
                #if not isinstance(processor, dict):
                    #module.set_processor(processor)
                #else:
                    #module.set_processor(processor.pop(f"{name}.processor"))

            #for sub_name, child in module.named_children():
                #fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        #for name, module in self.named_children():
            #fn_recursive_attn_processor(name, module, processor)

    ## Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    #def fuse_qkv_projections(self):
        #"""
        #Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        #are fused. For cross-attention modules, key and value projection matrices are fused.

        #<Tip warning={true}>

        #This API is 🧪 experimental.

        #</Tip>
        #"""
        #self.original_attn_processors = None

        #for _, attn_processor in self.attn_processors.items():
            #if "Added" in str(attn_processor.__class__.__name__):
                #raise ValueError(
                    #"`fuse_qkv_projections()` is not supported for models having added KV projections."
                #)

        #self.original_attn_processors = self.attn_processors

        #for module in self.modules():
            #if isinstance(module, Attention):
                #module.fuse_projections(fuse=True)

        #self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    ## Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    #def unfuse_qkv_projections(self):
        #"""Disables the fused QKV projection if enabled.

        #<Tip warning={true}>

        #This API is 🧪 experimental.

        #</Tip>

        #"""
        #if self.original_attn_processors is not None:
            #self.set_attn_processor(self.original_attn_processors)

    #def forward(
        #self,
        #hidden_states: torch.Tensor,
        #encoder_hidden_states: torch.Tensor,
        #timestep: Union[int, float, torch.LongTensor],
        #timestep_cond: Optional[torch.Tensor] = None,
        #ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        #image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        #attention_kwargs: Optional[Dict[str, Any]] = None,
        #return_dict: bool = True,
    #):
        #if attention_kwargs is not None:
            #attention_kwargs = attention_kwargs.copy()
            #lora_scale = attention_kwargs.pop("scale", 1.0)
        #else:
            #lora_scale = 1.0

        #if USE_PEFT_BACKEND:
            ## weight the lora layers by setting `lora_scale` for each PEFT layer
            #scale_lora_layers(self, lora_scale)
        #else:
            #if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                #logger.warning(
                    #"Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                #)

        #batch_size, num_frames, channels, height, width = hidden_states.shape

        ## 1. Time embedding
        #timesteps = timestep
        #t_emb = self.time_proj(timesteps)

        ## timesteps does not contain any weights and will always return f32 tensors
        ## but time_embedding might actually be running in fp16. so we need to cast here.
        ## there might be better ways to encapsulate this.
        #t_emb = t_emb.to(dtype=hidden_states.dtype)
        #emb = self.time_embedding(t_emb, timestep_cond)

        #if self.ofs_embedding is not None:
            #ofs_emb = self.ofs_proj(ofs)
            #ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            #ofs_emb = self.ofs_embedding(ofs_emb)
            #emb = emb + ofs_emb

        ## 2. Patch embedding
        #hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        #hidden_states = self.embedding_dropout(hidden_states)

        #text_seq_length = encoder_hidden_states.shape[1]
        #encoder_hidden_states = hidden_states[:, :text_seq_length]
        #hidden_states = hidden_states[:, text_seq_length:]

        ## 3. Transformer blocks
        #for i, block in enumerate(self.transformer_blocks):
            #if torch.is_grad_enabled() and self.gradient_checkpointing:

                #def create_custom_forward(module):
                    #def custom_forward(*inputs):
                        #return module(*inputs)

                    #return custom_forward

                #ckpt_kwargs: Dict[str, Any] = (
                    #{"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                #)
                #hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    #create_custom_forward(block),
                    #hidden_states,
                    #encoder_hidden_states,
                    #emb,
                    #image_rotary_emb,
                    #**ckpt_kwargs,
                #)
            #else:
                #hidden_states, encoder_hidden_states = block(
                    #hidden_states=hidden_states,
                    #encoder_hidden_states=encoder_hidden_states,
                    #temb=emb,
                    #image_rotary_emb=image_rotary_emb,
                #)

        #if not self.config.use_rotary_positional_embeddings:
            ## CogVideoX-2B
            #hidden_states = self.norm_final(hidden_states)
        #else:
            ## CogVideoX-5B
            #hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            #hidden_states = self.norm_final(hidden_states)
            #hidden_states = hidden_states[:, text_seq_length:]

        ## 4. Final block
        #hidden_states = self.norm_out(hidden_states, temb=emb)
        #hidden_states = self.proj_out(hidden_states)

        ## 5. Unpatchify
        #p = self.config.patch_size
        #p_t = self.config.patch_size_t

        #if p_t is None:
            #output = hidden_states.reshape(
                #batch_size, num_frames, height // p, width // p, -1, p, p
            #)
            #output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        #else:
            #output = hidden_states.reshape(
                #batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            #)
            #output = (
                #output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
            #)

        #if USE_PEFT_BACKEND:
            ## remove `lora_scale` from each PEFT layer
            #unscale_lora_layers(self, lora_scale)

        #if not return_dict:
            #return (output,)
        #return Transformer2DModelOutput(sample=output)


#if __name__ == "__main__":
    #model = CogVideoXTransformer3DModel()
    #print(model)
    #test_input = torch.randn(1, 49, 16, 60, 90)
    #test_encoder_hidden_states = torch.randn(1, 226, 16, 60, 90)
    #test_timestep = 0
    #output = model(test_input, test_encoder_hidden_states, test_timestep)

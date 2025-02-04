import decord
import torch
import json

from diffusers import (
    CogVideoXTransformer3DModel,
)
from finetune.models.transformer import CogVideoXTransformer3DActionModel, config_5b, config_2b, config_2b_iv

import decord
import torch
import json

from diffusers import (
    CogVideoXTransformer3DModel,
)
from finetune.models.transformer import CogVideoXTransformer3DActionModel, config_5b, config_2b, config_2b_iv


model_path = "THUDM/CogVideoX1.5-5B-I2V"
transformer_weights = CogVideoXTransformer3DModel.from_pretrained(
   model_path, subfolder="transformer"
)
transformer = CogVideoXTransformer3DActionModel(**config_5b)
transformer.load_state_dict(transformer_weights.state_dict(), strict=False)
transformer.save_pretrained("outputs/transformer_5b")
reloaded_model = CogVideoXTransformer3DActionModel.from_pretrained("outputs/transformer_5b")

for key in transformer_weights.state_dict().keys():
    if not torch.equal(transformer_weights.state_dict()[key], reloaded_model.state_dict()[key]):
        print(f"Mismatch found in layer: {key}")


model_path = "THUDM/CogVideoX-2b"
transformer_weights = CogVideoXTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer"
)
transformer = CogVideoXTransformer3DActionModel(**config_2b_iv)
state_dict = transformer_weights.state_dict()
model_state_dict = transformer.state_dict()

filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
transformer.load_state_dict(filtered_state_dict, strict=False)
transformer.save_pretrained("outputs/transformer_2b_iv")

# check whether the parameters agree
reloaded_model = CogVideoXTransformer3DActionModel.from_pretrained("outputs/transformer_2b_iv")
for key in transformer_weights.state_dict().keys():
    if not torch.equal(transformer_weights.state_dict()[key], reloaded_model.state_dict()[key]):
        print(f"Mismatch found in layer: {key}")


#open_json_and_add_config("outputs/transformer/config.json", config_2b)

model = CogVideoXTransformer3DActionModel.from_pretrained("outputs/transformer_2b_iv")
print(0)

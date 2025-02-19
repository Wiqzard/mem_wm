import decord
#print(200* "decord deactivated")
import sys
from pathlib import Path
import multiprocessing
import os
import random
#import torch
#import numpy as np
#import torch.distributed as dist





sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args

#def seed_everything(seed):
    #rank = dist.get_rank() if dist.is_initialized() else 0
    #seed = seed + rank  # Offset seed by process rank
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    
    #if torch.cuda.is_available():
        #torch.cuda.manual_seed_all(seed)

## Call this at the beginning of your script



def main():
    #multiprocessing.set_start_method("spawn", force=True)
    #seed_everything(42)  # Use any base seed
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()

print(10)

import sys
from pathlib import Path



sys.path.append(str(Path(__file__).parent.parent))

#from finetune.models.utils import get_model_cls
from finetune.schemas import Args
from finetune.online_training import Trainer
print(11)

#import os
#os.environ["NCCL_DEBUG"] = "INFO"

def main():
    args = Args.parse_args()
    #trainer_cls = get_model_cls(args.model_name, args.training_type)
    print(1)
    trainer = Trainer(args)
    print(2)
    trainer.fit()


if __name__ == "__main__":
    main()

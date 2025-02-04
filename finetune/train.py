import decord
import sys
from pathlib import Path
import multiprocessing



sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args


def main():
    #multiprocessing.set_start_method("spawn", force=True)

    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()

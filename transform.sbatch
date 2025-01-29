#!/bin/bash
#SBATCH --job-name=dw
#SBATCH --output=download_data_%j.log
#SBATCH --error=download_data_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --environment=gem
#SBATCH --container-workdir=/users/sstapf/test_data



#module load python  # Load the required Python module if needed
#srun python download_conc.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d" --json-files "/users/sstapf/test_data/build-house-Jul-28.json"
#python download_conc.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d" --json-files "/users/sstapf/test_data/build-house-Jul-28.json"

#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/10" --json-files "/users/sstapf/test_data/build-house-Jul-28.json"
# -----------------
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/11" --json-file "/users/sstapf/test_data/find-cave-Jul-28.json"
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/12" --json-file "/users/sstapf/test_data/waterfall-Jul-28.json"
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/13" --json-file "/users/sstapf/test_data/pen-animals-Jul-28.json"
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/14" --json-file "/users/sstapf/test_data/build-house-Jul-28.json"
#
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/8" --json-file "/users/sstapf/test_data/all_8xx_Jun_29.json"
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/9" --json-file "/users/sstapf/test_data/all_9xx_Jun_29.json"
#srun python download_dataset.py --output-dir="/capstor/store/cscs/swissai/a03/datasets/ego2d/10" --json-file "/users/sstapf/test_data/all_10xx_Jun_29.json"


#srun python transform.py
srun python transform_eval.py

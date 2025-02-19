from huggingface_hub import hf_hub_download, snapshot_download


snapshot_download(repo_id="KwaiVGI/GameFactory-Dataset", repo_type="dataset", 
cache_dir="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc",
local_dir="/capstor/store/cscs/swissai/a03/datasets/ego4d_mc")



from huggingface_hub import snapshot_download

repo_id = "qbo-odp/sift1m"
local_dir = "./sift1m_data"

snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset")
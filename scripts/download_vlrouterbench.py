"""Download and unpack VL-RouterBench dataset."""
import os
import sys
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    dest_dir = Path("F:/ADI/SDP/data/vl_routerbench")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "vlm_router_data.tar.gz"

    if not tar_path.exists():
        print(f"Downloading vlm_router_data.tar.gz from HuggingFace KinghtH/VL-RouterBench...")
        downloaded = hf_hub_download(
            repo_id="KinghtH/VL-RouterBench",
            repo_type="dataset",
            filename="vlm_router_data.tar.gz",
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to {downloaded}")
    else:
        print(f"Archive already exists at {tar_path}")

    # Extract
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    print(f"Extraction complete in {dest_dir}!")

if __name__ == "__main__":
    main()


import os
from pathlib import Path

from oot_diffusion.inference_segmentation import ClothesMaskModel


DEFAULT_HG_ROOT = Path(os.getcwd()) / "oodt_models"


def run_segmentation():
    cmm = ClothesMaskModel(
        hg_root="/content/models2",
        cache_dir="/content/drive/MyDrive/hf_cache",
    )

    result = cmm.generate("/content/model_1.png")


if __name__ == "__main__":
    run_segmentation()

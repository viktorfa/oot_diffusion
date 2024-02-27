import os
from PIL import Image
from pathlib import Path
from huggingface_hub import snapshot_download
import time
import torch


from oot_diffusion.humanparsing.aigc_run_parsing import Parsing
from oot_diffusion.ootd_utils import get_mask_location, resize_crop_center
from oot_diffusion.openpose.run_openpose import OpenPose


_category_get_mask_input = {
    "upperbody": "upper_body",
    "lowerbody": "lower_body",
    "dress": "dresses",
}

DEFAULT_HG_ROOT = Path(os.getcwd()) / "oodt_models"


class ClothesMaskModel:
    def __init__(self, hg_root: str = None, cache_dir: str = None):
        """
        Args:
            hg_root (str, optional): Path to the hg root directory. Defaults to CWD.
            cache_dir (str, optional): Path to the cache directory. Defaults to None.
        """
        if hg_root is None:
            hg_root = DEFAULT_HG_ROOT
        self.hg_root = hg_root
        self.cache_dir = cache_dir

        SEGMENTATION_PATH = f"{self.hg_root}/checkpoints/humanparsing"
        OPENPOSE_PATH = f"{self.hg_root}/checkpoints/openpose"

        if not Path(SEGMENTATION_PATH).exists() or not Path(OPENPOSE_PATH).exists():
            print("Downloading segmentation models")
            snapshot_download(
                "levihsu/OOTDiffusion",
                cache_dir=cache_dir,
                local_dir=hg_root,
                allow_patterns=["**/humanparsing/**", "**/openpose/**"],
            )

    def generate(
        self,
        model_path: str | bytes | Path | Image.Image,
    ):
        return self.generate_static(
            model_path,
            self.hg_root,
        )

    @staticmethod
    def generate_static(
        model_path: str | bytes | Path | Image.Image,
        hg_root: str = None,
    ):
        if hg_root is None:
            hg_root = DEFAULT_HG_ROOT

        category = "upperbody"

        if isinstance(model_path, Image.Image):
            model_image = model_path
        else:
            model_image = Image.open(model_path)

        model_image = resize_crop_center(model_image, 384, 512)

        start_model_parse = time.perf_counter()

        model_parse, face_mask = Parsing(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            hg_root,
        )(model_image)
        end_model_parse = time.perf_counter()
        print(f"Model parse in {end_model_parse - start_model_parse:.2f} seconds.")
        start_open_pose = time.perf_counter()

        keypoints = OpenPose(hg_root)(model_image)
        end_open_pose = time.perf_counter()
        print(f"Open pose in {end_open_pose - start_open_pose:.2f} seconds.")
        mask, mask_gray = get_mask_location(
            "hd",
            _category_get_mask_input[category],
            model_parse,
            keypoints,
            width=384,
            height=512,
        )
        mask = mask
        mask_gray = mask_gray

        masked_vton_img = Image.composite(mask_gray, model_image, mask)
        masked_vton_img = masked_vton_img.convert("RGB")

        return (
            masked_vton_img,
            mask,
            model_image,
            model_parse,
            Image.fromarray(face_mask * 255),
        )

import os
from PIL import Image
from pathlib import Path
import time
import torch

from oot_diffusion.humanparsing.inference import BodyParsingModel
from oot_diffusion.ootd_utils import get_mask_location, resize_crop_center
from oot_diffusion.openpose.inference import PoseModel


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        start_model_parse_load = time.perf_counter()
        self.human_parsing_model = BodyParsingModel(
            device=self.device,
            hg_root=self.hg_root,
            cache_dir=self.cache_dir,
        )
        self.human_parsing_model.load_model()
        end_model_parse_load = time.perf_counter()
        print(
            f"Model parse load in {end_model_parse_load - start_model_parse_load:.2f} seconds."
        )

        start_pose_load = time.perf_counter()
        self.pose_model = PoseModel(
            hg_root=self.hg_root,
            cache_dir=self.cache_dir,
        )
        self.pose_model.load_pose_model()
        end_pose_load = time.perf_counter()
        print(f"Pose load in {end_pose_load - start_pose_load:.2f} seconds.")

    def generate(
        self,
        model_path: str | bytes | Path | Image.Image,
    ):
        return self.generate_static(
            model_path=model_path,
            human_parsing_model=self.human_parsing_model,
            pose_model=self.pose_model,
            hg_root=self.hg_root,
        )

    @staticmethod
    def generate_static(
        model_path: str | bytes | Path | Image.Image,
        human_parsing_model: BodyParsingModel,
        pose_model: PoseModel,
        hg_root: str = None,
    ):
        if hg_root is None:
            hg_root = DEFAULT_HG_ROOT

        category = "upperbody"

        if isinstance(model_path, Image.Image):
            model_image = model_path
        else:
            model_image = Image.open(model_path)

        model_image = resize_crop_center(model_image, 384, 512).convert("RGB")

        start_model_parse = time.perf_counter()

        model_parse, _, face_mask = human_parsing_model.infer_parse_model(model_image)
        end_model_parse = time.perf_counter()
        print(f"Model parse in {end_model_parse - start_model_parse:.2f} seconds.")

        start_open_pose = time.perf_counter()

        keypoints = pose_model.infer_keypoints(model_image)
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
            face_mask,
        )

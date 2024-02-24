import os
from PIL import Image
from pathlib import Path


from .humanparsing.aigc_run_parsing import Parsing
from .inference_ootd import OOTDiffusion
from .ootd_utils import get_mask_location
from .openpose.run_openpose import OpenPose


_category_get_mask_input = {
    "upperbody": "upper_body",
    "lowerbody": "lower_body",
    "dress": "dresses",
}

DEFAULT_HG_ROOT = Path(os.getcwd()) / "oodt_models"


class OOTDiffusionModel:
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

    def load_pipe(self):
        self.pipe = OOTDiffusion(
            hg_root=self.hg_root,
            cache_dir=self.cache_dir,
        )
        return self.pipe

    def get_pipe(self):
        if not hasattr(self, "pipe"):
            self.load_pipe()
        return self.pipe

    def generate(
        self,
        cloth_path: str | bytes | Path | Image.Image,
        model_path: str | bytes | Path | Image.Image,
        seed=0,
        steps=10,
        cfg=2.0,
        num_samples=1,
    ):
        return self.generate_static(
            self.get_pipe(),
            cloth_path,
            model_path,
            self.hg_root,
            seed,
            steps,
            cfg,
            num_samples,
        )

    @staticmethod
    def generate_static(
        pipe,
        cloth_path: str | bytes | Path | Image.Image,
        model_path: str | bytes | Path | Image.Image,
        hg_root: str = None,
        seed=0,
        steps=10,
        cfg=2.0,
        num_samples=1,
    ):
        if hg_root is None:
            hg_root = DEFAULT_HG_ROOT

        category = "upperbody"

        if isinstance(cloth_path, Image.Image):
            cloth_image = cloth_path
        else:
            cloth_image = Image.open(cloth_path)
        if isinstance(model_path, Image.Image):
            model_image = model_path
        else:
            model_image = Image.open(model_path)
        model_image = model_image.resize((768, 1024))
        cloth_image = cloth_image.resize((768, 1024))

        model_parse, _ = Parsing(pipe.device, hg_root)(model_image.resize((384, 512)))
        keypoints = OpenPose()(model_image.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            pipe.model_type,
            _category_get_mask_input[category],
            model_parse,
            keypoints,
            width=384,
            height=512,
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_image, mask)
        images = pipe(
            category=category,
            image_garm=cloth_image,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=model_image,
            num_samples=num_samples,
            num_steps=steps,
            image_scale=cfg,
            seed=seed,
        )

        masked_vton_img = masked_vton_img.convert("RGB")

        return (images, masked_vton_img)

    def __str__(self):
        return str(self.pipe if hasattr(self, "pipe") else "OOTDiffusionModel")

    def __repr__(self):
        return str(self.pipe if hasattr(self, "pipe") else "OOTDiffusionModel")

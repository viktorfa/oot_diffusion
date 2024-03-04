import os
from PIL import Image, ImageOps
from pathlib import Path
from huggingface_hub import snapshot_download

from .inference_ootd import OOTDiffusion
from .ootd_utils import resize_crop_center


DEFAULT_HG_ROOT = Path(os.getcwd()) / "oodt_models"


class OOTDiffusionWithMaskModel:
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
        VAE_PATH = f"{self.hg_root}/checkpoints/ootd"
        if not Path(VAE_PATH).exists():
            print("Downloading VAE model")
            snapshot_download(
                "levihsu/OOTDiffusion",
                cache_dir=self.cache_dir,
                local_dir=self.hg_root,
                allow_patterns=["**/ootd/**"],
            )

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
        model_mask_path: str | bytes | Path | Image.Image,
        seed: int = 0,
        steps: int = 10,
        cfg: float = 2.0,
        num_samples: int = 1,
    ) -> list[Image.Image]:
        return self.generate_static(
            self.get_pipe(),
            cloth_path,
            model_path,
            model_mask_path,
            self.hg_root,
            seed,
            steps,
            cfg,
            num_samples,
        )

    @staticmethod
    def generate_static(
        pipe: OOTDiffusion,
        cloth_path: str | bytes | Path | Image.Image,
        model_path: str | bytes | Path | Image.Image,
        model_mask_path: str | bytes | Path | Image.Image,
        hg_root: str = None,
        seed: int = 0,
        steps: int = 10,
        cfg: float = 2.0,
        num_samples: int = 1,
    ) -> list[Image.Image]:
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
        if isinstance(model_mask_path, Image.Image):
            model_mask_image = model_mask_path
        else:
            model_mask_image = Image.open(model_mask_path)

        model_image = resize_crop_center(model_image, 768, 1024).convert("RGB")
        cloth_image = resize_crop_center(cloth_image, 768, 1024).convert("RGB")
        model_mask_image = (
            model_mask_image.resize((768, 1024), Image.LANCZOS)
            .convert("RGB")
            .convert("L")
        )

        gray_image = Image.new("L", model_image.size, 127)
        # Create an RGBA version of the original image
        original_rgba = model_image.convert("RGBA")

        # Create an RGBA version of the gray image
        # The alpha channel is the inverted binary mask where the masked areas are 0 (transparent)
        gray_rgba = Image.merge(
            "RGBA",
            (
                gray_image,
                gray_image,
                gray_image,
                ImageOps.invert(model_mask_image.convert("L")),
            ),
        )

        # Composite the images together using the binary mask as the alpha mask
        masked_vton_img = Image.composite(gray_rgba, original_rgba, model_mask_image)
        masked_vton_img = masked_vton_img.convert("RGB")

        images = pipe(
            category=category,
            image_garm=cloth_image,
            image_vton=masked_vton_img,
            mask=model_mask_image,
            image_ori=model_image,
            num_samples=num_samples,
            num_steps=steps,
            image_scale=cfg,
            seed=seed,
        )

        return images

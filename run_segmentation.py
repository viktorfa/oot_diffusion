import os
from pathlib import Path
import argparse

from oot_diffusion.inference_segmentation import ClothesMaskModel


DEFAULT_HG_ROOT = Path(os.getcwd()) / "ootd_models"
example_model_path = Path(__file__).parent / "oot_diffusion/assets/model_1.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oms diffusion")
    parser.add_argument("--person_path", type=str, default=str(example_model_path))
    parser.add_argument("--hg_root", type=str, default=str(DEFAULT_HG_ROOT))
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--output_path", type=str, default="./output_img")

    args = parser.parse_args()

    if args.person_path == str(example_model_path):
        print(
            f"Using example model image from {example_model_path}. Use --person_path to specify a different image."
        )
    if args.hg_root == str(DEFAULT_HG_ROOT):
        print(
            f"Using default hg_root to store models path {DEFAULT_HG_ROOT}. Use --hg_root to specify a different path."
        )

    cmm = ClothesMaskModel(
        hg_root=args.hg_root,
        cache_dir=args.cache_dir,
    )

    (masked_vton_img, mask, model_image, model_parse, face_mask) = cmm.generate(
        model_path=args.person_path
    )

    # Save files
    os.makedirs(args.output_path, exist_ok=True)
    with open(f"{args.output_path}/masked_vton_img.png", "wb") as f:
        masked_vton_img.save(f, "PNG")
    with open(f"{args.output_path}/mask.png", "wb") as f:
        mask.save(f, "PNG")
    with open(f"{args.output_path}/model_image.png", "wb") as f:
        model_image.save(f, "PNG")
    with open(f"{args.output_path}/model_parse.png", "wb") as f:
        model_parse.save(f, "PNG")
    with open(f"{args.output_path}/face_mask.png", "wb") as f:
        face_mask.save(f, "PNG")
    print(f"Saved files to {args.output_path}")

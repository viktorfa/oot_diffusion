import os
import argparse
from PIL import Image
from pathlib import Path
import time

from oot_diffusion.inference_with_mask import OOTDiffusionWithMaskModel

DEFAULT_HG_ROOT = Path(os.getcwd()) / "ootd_models"
example_model_path = Path(__file__).parent / "oot_diffusion/assets/model_1.png"
example_garment_path = Path(__file__).parent / "oot_diffusion/assets/cloth_1.jpg"
example_person_mask_path = (
    Path(__file__).parent / "oot_diffusion/assets/model_1_mask.png"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="oms diffusion")
    parser.add_argument("--cloth_path", type=str, default=str(example_garment_path))
    parser.add_argument("--person_path", type=str, default=str(example_model_path))
    parser.add_argument("--person_mask_path", type=str, default=str(example_model_path))
    parser.add_argument("--hg_root", type=str, default=str(DEFAULT_HG_ROOT))
    parser.add_argument("--cache_dir", type=str, required=False)
    parser.add_argument("--output_path", type=str, default="./output_img")

    args = parser.parse_args()

    if args.person_path == str(example_model_path):
        print(
            f"Using example model image from {example_model_path}. Use --person_path to specify a different image."
        )
    if args.cloth_path == str(example_garment_path):
        print(
            f"Using example garment image from {example_garment_path}. Use --cloth_path to specify a different image."
        )
    if args.person_mask_path == str(example_person_mask_path):
        print(
            f"Using example person mask image from {example_person_mask_path}. Use --person_mask_path to specify a different image."
        )
    if args.hg_root == str(DEFAULT_HG_ROOT):
        print(
            f"Using default hg_root to store models path {DEFAULT_HG_ROOT}. Use --hg_root to specify a different path."
        )

    model = OOTDiffusionWithMaskModel(
        hg_root=args.hg_root,
        cache_dir=args.cache_dir,
    )

    start_time = time.perf_counter()
    model.load_pipe()
    end_time_load_model = time.perf_counter()
    print(f"Model loaded in {end_time_load_model - start_time:.2f} seconds.")

    model_image = Image.open(args.person_path)
    garment_image = Image.open(args.cloth_path)
    person_mask_image = Image.open(args.person_mask_path)

    start_generate_time = time.perf_counter()
    result_images = model.generate(
        model_path=model_image,
        cloth_path=garment_image,
        model_mask_path=person_mask_image,
    )
    end_generate_time = time.perf_counter()
    print(f"Generated in {end_generate_time - start_generate_time:.2f} seconds.")

    os.makedirs(args.output_path, exist_ok=True)

    for i, result_image in enumerate(result_images):
        with open(f"{args.output_path}/result_image_{i}.png", "wb") as f:
            result_image.save(f, "PNG")

    print(f"See {args.output_path}/ for the result images.")

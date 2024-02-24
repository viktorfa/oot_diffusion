from oot_diffusion import OOTDiffusionModel
from PIL import Image
from pathlib import Path

hg_root_path = Path(__file__).parent / "ootd_models"
cache_dir_path = Path(__file__).parent / ".hf_cache"


if __name__ == "__main__":
    model = OOTDiffusionModel(
        hg_root=str(hg_root_path),
        cache_dir=str(cache_dir_path),
    )
    model.load_pipe()

    example_model_path = Path(__file__).parent / "oot_diffusion/assets/model_1.png"
    example_garment_path = Path(__file__).parent / "oot_diffusion/assets/cloth_1.jpg"

    model_image = Image.open(example_model_path)
    garment_image = Image.open(example_garment_path)

    result_images, result_mask = model.generate(
        model_path=model_image, cloth_path=garment_image
    )

    with open("output_images/result_mask.png", "wb") as f:
        result_mask.save(f, "PNG")
    for i, result_image in enumerate(result_images):
        with open(f"output_images/result_image_{i}.png", "wb") as f:
            result_image.save(f, "PNG")

    print("See output_images/ for the result images.")

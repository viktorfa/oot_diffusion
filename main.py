from oot_diffusion import OOTDiffusionModel
from PIL import Image
from pathlib import Path

hg_root_path = Path(__file__).parent / "models"
cache_dir_path = Path(__file__).parent / ".hf_cache"

model = OOTDiffusionModel(
    hg_root=str(hg_root_path),
    cache_dir=str(cache_dir_path),
)


if __name__ == "__main__":
    print(model)
    print(model.load())

    example_model_path = Path(__file__).parent / "assets/model_1.png"
    example_garment_path = Path(__file__).parent / "assets/cloth_1.jpg"

    model_image = Image.open(example_model_path)
    garment_image = Image.open(example_garment_path)

    print(model.generate(model, model_image, garment_image))

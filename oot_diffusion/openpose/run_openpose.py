import numpy as np


# from pytorch_lightning import seed_everything
from .annotator.util import resize_image, HWC3
from .annotator.openpose import OpenposeDetector

from PIL import Image
import torch


class OpenPose:
    def __init__(self, hg_root: str):
        self.preprocessor = OpenposeDetector(hg_root=hg_root)

    def __call__(self, input_image, resolution=384):
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif type(input_image) == str:
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            assert H == 512 and W == 384, "Incorrect input image shape"
            pose = self.preprocessor(input_image, return_is_index=True)

            candidate = pose["bodies"]["candidate"]
            subset = pose["bodies"]["subset"][0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if (subset[j]) != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if (subset[j]) != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            for i in range(18):
                candidate[i][0] *= 384
                candidate[i][1] *= 512

            keypoints = {"pose_keypoints_2d": candidate}

        return keypoints

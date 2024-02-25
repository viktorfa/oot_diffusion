# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from .body import Body


class OpenposeDetector:
    def __init__(self, hg_root: str):
        body_modelpath = os.path.join(
            hg_root,
            "checkpoints/openpose/ckpts",
            "body_pose_model.pth",
        )

        self.body_estimation = Body(body_modelpath)

    def __call__(self, oriImg):
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            hands = []
            faces = []

            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            return pose

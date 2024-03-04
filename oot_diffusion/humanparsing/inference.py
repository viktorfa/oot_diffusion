from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import numpy as np

from oot_diffusion.humanparsing.utils import find_midpoint_y, label_map, get_palette


class BodyParsingModel:
    def __init__(self, device: str = "cpu", hg_root: str = None, cache_dir: str = None):
        self.hg_root = hg_root
        self.cache_dir = cache_dir
        self.device = device

    def load_model(self):
        self.processor_face = SegformerImageProcessor.from_pretrained(
            "jonathandinu/face-parsing",
            cache_dir=self.cache_dir,
        )
        self.model_face = AutoModelForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing",
            cache_dir=self.cache_dir,
        )
        self.model_face.to(self.device)

        self.processor_clothes = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            cache_dir=self.cache_dir,
        )
        self.model_clothes = AutoModelForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            cache_dir=self.cache_dir,
        )
        self.model_clothes.to(self.device)

    def infer_parse_model(self, image: Image.Image):
        inputs_face = self.processor_face(images=image, return_tensors="pt").to(
            self.device
        )

        outputs_face = self.model_face(**inputs_face)
        logits_face = outputs_face.logits.cpu()

        upsampled_logits_face = nn.functional.interpolate(
            logits_face,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg_face = upsampled_logits_face.argmax(dim=1)[0]

        inputs_clothes = self.processor_clothes(images=image, return_tensors="pt").to(
            self.device
        )

        outputs_clothes = self.model_clothes(**inputs_clothes)
        logits_clothes = outputs_clothes.logits.cpu()

        upsampled_logits_clothes = nn.functional.interpolate(
            logits_clothes,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg_clothes = upsampled_logits_clothes.argmax(dim=1)[0]

        image, segmentation, face_image = process_segmentation(
            pred_seg_clothes, pred_seg_face
        )

        palette = get_palette(19)
        image.putpalette(palette)

        return image, segmentation, face_image


def process_segmentation(pred_seg_clothes, pred_seg_face):
    # Convert segmentation to numpy array for processing
    segmentation_body = pred_seg_clothes.numpy().astype(np.uint8)
    segmentation_face = pred_seg_face.numpy().astype(np.uint8)

    # Define class indices (adjust these based on your model's classes)
    neck_index = 18  # The new class index for the neck
    face_index_body = label_map["head"]  # The class index for face in body segmentation
    neck_index_face = 17  # The class index for neck in face segmentation

    # Create a neck mask from the face segmentation
    neck_mask_face = segmentation_face == neck_index_face
    midpoint = find_midpoint_y(neck_mask_face)

    # Create an upper body skin mask from the body segmentation, excluding the face
    body_face_mask = segmentation_body == face_index_body

    # Combine the neck mask and the upper body skin mask
    only_neck_mask = np.logical_and(body_face_mask, neck_mask_face)

    # Apply the combined mask to the body segmentation
    segmentation_body[only_neck_mask] = neck_index

    face_mask = np.zeros_like(segmentation_body)
    if midpoint is not None:
        # Mark pixels as "neck" if they are "face" in body segmentation and below the neck midpoint
        face_pixels_below_midpoint = segmentation_body[midpoint:] == face_index_body
        segmentation_body[midpoint:][face_pixels_below_midpoint] = neck_index
        face_mask = np.isin(segmentation_body, [1, 2, 3, 11])

    # Convert the modified segmentation back to a PIL Image for visualization
    output_img = Image.fromarray(segmentation_body)
    output_img_face = Image.fromarray(face_mask)

    return output_img, segmentation_body, output_img_face

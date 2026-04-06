import os
import torch
import io
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from diffusers import StableDiffusionInpaintPipeline

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

seg_id = "facebook/mask2former-swin-tiny-ade-semantic"
processor = AutoImageProcessor.from_pretrained(seg_id, token=HF_TOKEN)
seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(seg_id, token=HF_TOKEN).to(device)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    token=HF_TOKEN,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

if device == "cpu":
    pipe.enable_attention_slicing()
else:
    pipe.to(device)


def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def detect_missing_regions(init_image):
    gray = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    inputs = processor(images=init_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)

    prediction = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[init_image.size[::-1]]
    )[0].cpu().numpy()

    seg_mask = np.where(
        (prediction == 0) | (prediction == 1) | (prediction == 255),
        255,
        0
    ).astype(np.uint8)

    combined = cv2.bitwise_or(edges, seg_mask)
    combined = refine_mask(combined)

    if np.sum(combined) < 3000:
        combined = np.ones_like(gray) * 255

    return combined


def reconstruct_artifact(image_bytes):
    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))

    mask_array = detect_missing_regions(init_image)

    coverage = np.sum(mask_array) / (512 * 512 * 255)
    if coverage < 0.02:
        return init_image

    mask_image = Image.fromarray(mask_array)

    prompt = (
        "infrared underpainting, hidden sketch beneath painting, "
        "classical artwork, faded pigments, original composition, "
        "historically accurate restoration, museum quality"
    )

    negative_prompt = (
        "modern objects, distorted shapes, unrealistic textures, "
        "blurry, oversmoothed, low quality"
    )

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=28,
            guidance_scale=8.0,
            strength=0.65
        ).images[0]

    return result
"""
Utility functions for tensor and PIL image conversions.
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Union, Optional


def tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor images to PIL Images.

    Args:
        images: Tensor of shape [B, H, W, C] with values in [0, 1]

    Returns:
        List of PIL Images

    Raises:
        ValueError: If input is not a torch.Tensor or has unsupported shape
    """
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(images)}")

    # Ensure tensor is on CPU and in correct format
    images = images.cpu()

    # Handle different tensor shapes
    if images.dim() == 3:
        # Single image [H, W, C]
        images = images.unsqueeze(0)
    elif images.dim() == 2:
        # Grayscale [H, W]
        images = images.unsqueeze(0).unsqueeze(-1)

    # Convert to [0, 255] range
    if images.max() <= 1.0:
        images = images * 255.0

    images = images.clamp(0, 255).byte()

    # Convert each image in batch to PIL
    pil_images = []
    for img in images:
        img_np = img.numpy()
        if img_np.shape[-1] == 1:
            # Grayscale
            pil_img = Image.fromarray(img_np.squeeze(-1), mode='L')
        elif img_np.shape[-1] == 3:
            # RGB
            pil_img = Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 4:
            # RGBA
            pil_img = Image.fromarray(img_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported channel count: {img_np.shape[-1]}")
        pil_images.append(pil_img)

    return pil_images


def pil_to_tensor(pil_images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
    """
    Convert PIL Images to tensor format.

    Args:
        pil_images: Single PIL Image or list of PIL Images

    Returns:
        Tensor of shape [B, H, W, C] with values in [0, 1]

    Raises:
        ValueError: If input is not a PIL Image or list of PIL Images
    """
    if isinstance(pil_images, Image.Image):
        pil_images = [pil_images]

    if not isinstance(pil_images, list):
        raise ValueError(f"Expected PIL Image or list of PIL Images, got {type(pil_images)}")

    tensor_list = []
    for pil_img in pil_images:
        if not isinstance(pil_img, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(pil_img)}")

        # Convert to RGB if needed
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert to numpy array
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np)
        tensor_list.append(img_tensor)

    # Stack into batch
    images_tensor = torch.stack(tensor_list)

    return images_tensor


def masks_to_tensor(masks: Union[torch.Tensor, Image.Image, List, np.ndarray]) -> Optional[torch.Tensor]:
    if isinstance(masks, torch.Tensor):
        # Ensure float type and range [0, 1]
        masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        return masks.cpu()
    elif isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
        if masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        return masks

    return masks

# code based on  https://github.com/PozzettiAndrea/ComfyUI-SAM3/blob/main/nodes/utils.py
def visualize_masks_on_image(image, masks, boxes=None, scores=None, alpha=0.5):
    """
    Create visualization of masks overlaid on image

    Args:
        image: PIL Image or numpy array
        masks: torch.Tensor [N, H, W] binary masks
        boxes: Optional torch.Tensor [N, 4] bounding boxes in [x0, y0, x1, y1]
        scores: Optional torch.Tensor [N] confidence scores
        alpha: Transparency of mask overlay

    Returns:
        PIL Image with visualization
    """
    if isinstance(image, torch.Tensor):
        image = pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))

    # Convert to numpy for processing
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize masks to image size if needed
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    # Create colored overlay
    np.random.seed(42)  # Consistent colors
    overlay = img_np.copy()

    for i, mask in enumerate(masks_np):
        # Squeeze extra dimensions (masks may be [1, H, W] or [H, W])
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        # Resize mask to image size if needed
        if mask.shape != img_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        # Random color for this mask
        color = np.random.rand(3)

        # Apply colored mask
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0.5,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    # Convert back to PIL
    result = Image.fromarray((overlay * 255).astype(np.uint8))

    # Draw boxes if provided
    if boxes is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)

        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes

        for i, box in enumerate(boxes_np):
            x0, y0, x1, y1 = box

            # Random color for this box (same seed for consistency)
            np.random.seed(42 + i)
            color_int = tuple((np.random.rand(3) * 255).astype(int).tolist())

            # Draw box
            draw.rectangle([x0, y0, x1, y1], outline=color_int, width=3)

            # Draw score if provided
            if scores is not None:
                score = scores[i] if isinstance(scores, (list, np.ndarray)) else scores[i].item()
                text = f"{score:.2f}"
                draw.text((x0, y0 - 15), text, fill=color_int)

    return result

def resize_mask(mask, shape):
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)

def join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor, invert=False):
    batch_size = min(len(image), len(alpha))
    out_images = []

    if invert:
        alpha = 1.0 - resize_mask(alpha, image.shape[1:])
    else:
        alpha = resize_mask(alpha, image.shape[1:])
    for i in range(batch_size):
        out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

    return torch.stack(out_images),
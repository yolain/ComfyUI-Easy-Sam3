# ComfyUI-Easy-Sam3

[English](README.md) | [中文](README_CN.md)

A ComfyUI custom node package for [SAM3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3), providing powerful image and video segmentation capabilities with text prompts.

## Overview

This node package brings Meta's SAM3 model to ComfyUI, enabling:
- **Image Segmentation**: Segment objects in images using text descriptions
- **Video Tracking**: Track and segment objects across video frames
- **Advanced Configuration**: Fine-tune video tracking parameters for optimal results

### Image Segmentation
![Image Segmentation Example](assets/image.png)
*Example of semantic segmentation on images*

### Video Segmentation
![Video Segmentation Example](assets/video.png)
*Example of semantic segmentation on video frames*

> **Note**: Currently, this package supports **semantic segmentation** only.

## Features

- 🖼️ **Image Segmentation**: Segment objects using natural language prompts
- 🎬 **Video Segmentation**: Track objects across video frames with consistent IDs
- 🎨 **Background Options**: Add custom backgrounds (black, white, grey) to segmented images
- ⚙️ **Flexible Configuration**: Support for different devices (CUDA, CPU, MPS) and precisions (fp32, fp16, bf16)
- 🔧 **Advanced Controls**: Comprehensive video tracking parameters for fine-tuning


## Nodes

### 1. Load SAM3 Model
Load a SAM3 model for image or video segmentation.

**Inputs:**
- `model`: SAM3 model file from the models/sam3 folder
- `segmentor`: Choose between "image" or "video" mode
- `device`: Device to load the model on (cuda, cpu, mps)
- `precision`: Model precision (fp32, fp16, bf16)

**Outputs:**
- `sam3_model`: Loaded SAM3 model for downstream nodes

### 2. SAM3 Image Segmentation
Segment objects in images using text prompts.

**Inputs:**
- `sam3_model`: SAM3 model from Load SAM3 Model node
- `images`: Input images to segment
- `prompt`: Text description of objects to segment (e.g., "a cat", "person")
- `threshold`: Confidence threshold for detections (0.0-1.0)
- `keep_model_loaded`: Keep model in VRAM after inference
- `add_background`: Add background color (none, black, white, grey)

**Outputs:**
- `masks`: Segmentation masks
- `images`: Segmented images (with optional background)

### 3. SAM3 Video Segmentation
Track and segment objects across video frames.

**Inputs:**
- `sam3_model`: SAM3 model in video mode
- `session_id`: Optional session ID to resume tracking
- `video_frames`: Video frames as image sequence
- `prompt`: Text description of objects to track
- `score_threshold_detection`: Detection confidence threshold
- `new_det_thresh`: Threshold for adding new objects
- `propagation_direction`: Propagation direction (both, forward, backward)
- `start_frame_index`: Frame index to start propagation
- `keep_model_loaded`: Keep model in VRAM
- `close_after_propagation`: Close session after completion
- `extra_config`: Additional configuration from Extra Config node

**Outputs:**
- `masks`: Tracked segmentation masks for all frames
- `session_id`: Session ID for resuming tracking

### 4. SAM3 Video Model Extra Config
Configure advanced parameters for video segmentation.

**Key Parameters:**
- `assoc_iou_thresh`: IoU threshold for detection-to-track matching
- `trk_assoc_iou_thresh`: Stricter IoU threshold for unmatched masklets
- `hotstart_delay`: Delay outputs to remove unmatched/duplicate tracklets
- `max_trk_keep_alive`: Maximum frames to keep track alive without detection
- `det_nms_thresh`: IoU threshold for NMS
- `fill_hole_area`: Fill holes in masks smaller than this area
- `max_num_objects`: Maximum number of objects to track
- And many more fine-tuning options...

**Output:**
- `extra_config`: Configuration dictionary for Video Segmentation node

## Usage Examples

### Basic Image Segmentation
1. Load SAM3 Model (mode: image)
2. Connect to SAM3 Image Segmentation
3. Provide input images and text prompt
4. Get segmentation masks and images

### Video Object Tracking
1. Load SAM3 Model (mode: video)
2. (Optional) Create Extra Config node for advanced settings
3. Connect to SAM3 Video Segmentation
4. Provide video frames and tracking parameters
5. Get tracked masks across all frames

## Model Downloads

Download SAM3 model weights from the official repository:
- [SAM3 Models](https://huggingface.co/facebook/sam3)

Place the downloaded models in: `ComfyUI/models/sam3/`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- ComfyUI
- CUDA-compatible GPU (recommended)

## Localization

This node package supports multiple languages:
- English (`locales/en/nodeDefs.json`)
- Chinese (`locales/zh/nodeDefs.json`)

## Credits

- **SAM3**: [Facebook Research](https://github.com/facebookresearch/sam3)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## License

This project follows the license of the original SAM3 repository.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Changelog

### v1.0.0
- Initial release
- Image segmentation with text prompts
- Video tracking and segmentation
- Background color options for image segmentation
- Advanced video tracking configuration
- Multi-language support (EN/ZH)

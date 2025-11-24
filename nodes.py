import os
import json
import folder_paths
import numpy as np
import torch
import comfy.model_management as mm
import comfy.utils
import logging

from contextlib import nullcontext

from PIL import Image
from typing import Tuple, Any
from comfy_api.latest import ComfyExtension, io, ui
from .sam3.logger import get_logger
from .utils import tensor_to_pil, pil_to_tensor, masks_to_tensor, join_image_with_alpha, parse_points, parse_bbox, draw_visualize_image

logger = get_logger(__name__)

# Register sam3 model folder path
if "sam3" not in folder_paths.folder_names_and_paths:
    sam3_models_dir = os.path.join(folder_paths.models_dir, "sam3")
    os.makedirs(sam3_models_dir, exist_ok=True)
    folder_paths.folder_names_and_paths["sam3"] = ([sam3_models_dir], folder_paths.supported_pt_extensions)

from .sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor


class LoadSam3Model(io.ComfyNode):
    """Load SAM3 model for image or video segmentation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy sam3ModelLoader",
            display_name="Load SAM3 Model",
            category="EasyUse/Sam3",
            description="Load SAM3 model for segmentation tasks",
            inputs=[
                io.Combo.Input(
                    "model",
                    options=folder_paths.get_filename_list("sam3"),
                    default="sam3.pt",
                    tooltip="Select SAM3 model file to load"
                ),
                io.Combo.Input(
                    "segmentor",
                    options=["image", "video"],
                    default="image",
                    tooltip="Choose between image or video segmentation mode"
                ),
                io.Combo.Input(
                    "device",
                    options=["cuda", "cpu", "mps"],
                    default="cuda",
                    tooltip="Device to load the model on"
                ),
                io.Combo.Input(
                    "precision",
                    options=["fp32", "fp16", "bf16"],
                    default="fp32",
                    tooltip="Model precision for inference"
                ),
                # io.Boolean.Input(
                #     "compile",
                #     default=False,
                #     tooltip="Compile the model for optimized performance"
                # ),
            ],
            outputs=[
                io.Custom(io_type="EASY_SAM3_MODEL").Output(display_name="sam3_model",)
            ]
        )

    @classmethod
    def execute(cls, model, segmentor, device, precision) -> io.NodeOutput:
        # Get model path
        model_path = folder_paths.get_full_path_or_raise("sam3", model)
        if model_path is None:
            raise ValueError(f"Model file '{model}' not found in sam3 folder")

        if "fp16" in model.lower():
            precision = "fp16"

        # Build model based on segmentor type
        if segmentor == "image":
            from .sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                checkpoint_path=model_path,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False
            )
            processor = Sam3Processor(
                model=model,
                resolution=1008,
                confidence_threshold=0.3
            )
        elif segmentor == "video":
            model = build_sam3_video_predictor(
                checkpoint_path=model_path,
                gpus_to_use=None
            )
            processor = None

        else:
            raise ValueError(f"Unknown segmentor type: {segmentor}")

        logger.info("Sam3 Model loaded successfully")

        if precision != 'fp32' and device == 'cpu':
            raise ValueError("fp16 and bf16 are not supported on cpu")

        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]

        sam3_model = {
            "model": model,
            "processor": processor,
            "segmentor": segmentor,
            "device": device,
            "dtype": dtype,
        }

        return io.NodeOutput(sam3_model)


class Sam3ImageSegmentation(io.ComfyNode):
    """Perform image segmentation using SAM3 model with text or geometric prompts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy sam3ImageSegmentation",
            display_name="Sam3 Image Segmentation",
            category="EasyUse/Sam3",
            description="Segment images using SAM3 with text prompts and optional box/point prompts",
            inputs=[
                io.Custom(io_type="EASY_SAM3_MODEL").Input(
                    "sam3_model",
                    display_name="SAM3 Model",
                    tooltip="SAM3 model loaded from LoadSam3Model node"
                ),
                io.Image.Input(
                    "images",
                    tooltip="Input image or images to segment"
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text description of objects to segment (e.g., 'a cat', 'person')"
                ),
                io.Float.Input(
                    "threshold",
                    default=0.40,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Confidence threshold for detections"
                ),
                io.Boolean.Input(
                    "keep_model_loaded",
                    default=False,
                ),
                io.Combo.Input(
                    "add_background",
                    options=["none", "black", "white", "grey"],
                    default="none",
                    tooltip="Add background color to segmented images"
                ),
                io.String.Input(
                    "coordinates_positive",
                    display_name="coordinates_positive",
                    optional=True,
                    force_input=True,
                ),
                io.String.Input(
                    "coordinates_negative",
                    display_name="coordinates_negative",
                    optional=True,
                    force_input=True,
                ),
                io.BBOX.Input(
                    "bboxes",
                    display_name="bboxes",
                    optional=True,
                ),
                io.Mask.Input(
                    "mask",
                    display_name="mask",
                    optional=True,
                ),
                io.Int.Input(
                    "detection_limit",
                    default=-1,
                    min=-1,
                    max=1000,
                    tooltip="Advanced: Limit number of detections (-1 for no limit)"
                )
            ],
            outputs=[
                io.Mask.Output(
                    "output_masks",
                    display_name="masks",
                    # is_output_list=True,
                    tooltip="Segmentation masks (combined per image)"
                ),
                io.Image.Output(
                    "output_images",
                    display_name="images",
                    tooltip="Segmentation images",
                ),
                io.Mask.Output(
                    "obj_masks",
                    display_name="obj_masks",
                    tooltip="Individual object masks before combining (for visualization)"
                ),
                io.BBOX.Output(
                    "boxes",
                    display_name="boxes",
                    tooltip="Bounding boxes for each detected object"
                ),
                io.Float.Output(
                    "scores",
                    display_name="scores",
                    tooltip="Confidence scores for each detected object"
                ),
            ]
        )

    @classmethod
    def execute(cls, sam3_model, images, prompt, threshold=0.3, keep_model_loaded=False, add_background='none', detection_limit=-1, coordinates_positive=None, coordinates_negative=None, bboxes=None, mask=None) -> io.NodeOutput:
        offload_device = mm.unet_offload_device()

        processor = sam3_model.get("processor", None)
        model = sam3_model.get("model", None)
        device = sam3_model.get("device", torch.device("cpu"))
        dtype = sam3_model.get("dtype", torch.float32)
        segmentor = sam3_model.get("segmentor", 'image')

        B, H, W, _ = images.shape

        if model is None or segmentor != "image":
            raise ValueError("Invalid SAM3 model. Please load a SAM3 model in 'image' mode")

        # set confidence threshold
        processor.set_confidence_threshold(threshold)

        # Parse inputs with bounds checking
        pos_points, pos_count, pos_errors = parse_points(coordinates_positive, images.shape)
        neg_points, neg_count, neg_errors = parse_points(coordinates_negative, images.shape)
        # Combine points for refinement
        points = None
        point_labels = None
        if pos_points is not None and neg_points is not None:
            points = pos_points + neg_points
            point_labels = [1] * pos_count + [0] * neg_count
        elif pos_points is not None:
            points = pos_points
            point_labels = [1] * pos_count
        elif neg_points is not None:
            points = neg_points
            point_labels = [0] * neg_count

        # bbox
        bounding_boxes = None
        bounding_box_labels = None
        if bboxes is not None:
            bbox_coords, bbox_count = parse_bbox(bboxes, images.shape)
            if bbox_coords is not None:
                bounding_boxes = bbox_coords
                bounding_box_labels = [True] * bbox_count

        # Switch model to main device
        model.to(device)
        if mask is not None:
            mask.to(device)

        autocast_condition = not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
             # Convert tensor images to PIL format for processor
            pil_images = tensor_to_pil(images)
            num_frames = len(pil_images)
            logger.info(f"Processing {num_frames} image(s)")

            # Process each image separately to maintain correspondence
            output_masks = []
            output_images = []
            output_boxes = []
            output_scores = []
            output_raw_masks = []

            # Initialize progress bar

            pbar = comfy.utils.ProgressBar(num_frames)
            processed_frames = 0

            for idx, pil_img in enumerate(pil_images):
                # Set single image in processor
                state = processor.set_image(pil_img)

                # text prompt
                prompt_text = prompt.strip()
                if prompt_text:
                    state = processor.set_text_prompt(prompt_text, state)

                # points
                if points is not None and len(points) > 0:
                    logging.info(f"Processing {len(points)} points")
                    state = processor.add_point_prompt(points, point_labels, state)
                # bbox
                if bounding_boxes is not None and len(bounding_boxes) > 0:
                    logger.info("Adding %d bounding box(es) as prompt", len(bounding_boxes))
                    state = processor.add_boxes_prompts(bounding_boxes, bounding_box_labels, state)
                # mask
                if mask is not None:
                    state = processor.add_mask_prompt(mask, state)

                # Get the masks and scores for this image
                masks = state.get('masks', None)
                boxes = state.get('boxes', None)
                scores = state.get('scores', None)

                # Handle empty results for this image
                if masks is None or len(masks) == 0:
                    logger.warning(f"No masks detected for image {idx}, using empty mask")
                    masks = torch.zeros(1, H, W)
                else:
                    # Sort by scores (highest confidence first)
                    if scores is not None and len(scores) > 0:
                        logger.info(f"Image {idx}: detected {len(masks)} mask(s) with top score: {scores.max().item():.3f}")
                        top_indices = torch.argsort(scores, descending=True)
                        masks = masks[top_indices]
                        boxes = boxes[top_indices]
                        scores = scores[top_indices]

                    if detection_limit > -1:
                        masks = masks[:detection_limit]
                        boxes = boxes[:detection_limit]
                        scores = scores[:detection_limit]

                output_raw_masks.append(masks)
                # Convert masks to tensor format
                masks_tensor = masks_to_tensor(masks)

                if masks_tensor is None or len(masks_tensor) == 0:
                    logger.warning(f"Failed to convert masks for image {idx}, using empty mask")
                    combined_mask = torch.zeros(1, H, W)
                else:
                    # Combine all masks for this image using logical OR (union of all detected objects)
                    # This creates a single mask that includes all detected objects
                    combined_mask = (masks_tensor.sum(dim=0) > 0).float()
                    logger.info(f"Image {idx}: combined {len(masks_tensor)} mask(s) into one")


                output_masks.append(combined_mask)

                img_tensor = pil_to_tensor(pil_img)
                mask_tensor = combined_mask.unsqueeze(0)
                rgba_image, = join_image_with_alpha(img_tensor, mask_tensor, False)

                if add_background != "none":
                    if add_background == "black":
                        bg_color = torch.zeros_like(rgba_image[:, :, :, :3])
                    elif add_background == "white":
                        bg_color = torch.ones_like(rgba_image[:, :, :, :3])
                    elif add_background == "grey":
                        bg_color = torch.ones_like(rgba_image[:, :, :, :3]) * 0.5

                    rgb = rgba_image[:, :, :, :3]
                    alpha = rgba_image[:, :, :, 3:4]

                    composited = rgb * alpha + bg_color * (1 - alpha)
                    output_images.append(composited.squeeze(0))
                else:
                    output_images.append(rgba_image.squeeze(0))

                output_boxes.append(boxes)
                output_scores.append(scores)

                # Update progress bar
                processed_frames += 1
                pbar.update_absolute(processed_frames, num_frames)

            output_masks = torch.stack(output_masks, dim=0)
            output_boxes = torch.stack(output_boxes, dim=0)
            output_scores = torch.stack(output_scores, dim=0)
            output_raw_masks = torch.stack(output_raw_masks, dim=0)
            logger.debug(f"Output masks shape: {output_masks.shape} (matches input images: {B})")

            # Clean up if not keeping model loaded
            if not keep_model_loaded:
                model.to(offload_device)
                mm.soft_empty_cache()

        return io.NodeOutput(output_masks, output_images,output_raw_masks, output_boxes, output_scores,)


class Sam3VideoSegmentation(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy sam3VideoSegmentation",
            display_name="SAM3 Video Segmentation",
            category="EasyUse/Sam3",
            description="Track and segment objects across video frames using SAM3",
            inputs=[
                io.Custom(io_type="EASY_SAM3_MODEL").Input(
                    "sam3_model",
                    display_name="SAM3 Model",
                    tooltip="SAM3 model loaded from LoadSam3Model node (must be video mode)"
                ),
                io.String.Input(
                    "session_id",
                    default=None,
                    force_input=True,
                    optional=True,
                ),
                io.Image.Input(
                    "video_frames",
                    tooltip="Video frames as image sequence"
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text description of objects to track (e.g., 'person', 'car')"
                ),
                io.Int.Input(
                    "frame_index",
                    min=0,
                    max=10 ** 5,
                    step=1,
                    tooltip="Frame where initial prompt is applied",
                ),
                io.Int.Input(
                    "object_id",
                    default=1,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Unique ID for multi-object tracking"
                ),
                io.Float.Input(
                    "score_threshold_detection",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Confidence threshold for detections, default is 0.5"
                ),
                io.Float.Input(
                    "new_det_thresh",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Threshold for a detection to be added as a new object, default is 0.7"
                ),
                io.Combo.Input(
                    "propagation_direction",
                    options=["both", "forward", "backward"],
                    default="both",
                ),
                io.Int.Input(
                    "start_frame_index",
                    default=0,
                    min=0,
                    max=10**5,
                    step=1,
                ),
                io.Int.Input(
                    "max_frames_to_track",
                    default=-1,
                    min=-1,
                    tooltip="Advanced: Max frames to process (-1 for all)"
                ),
                io.Boolean.Input(
                    "close_after_propagation",
                    default=True,
                    tooltip="Close the session after propagation"
                ),
                io.Boolean.Input(
                  "keep_model_loaded",
                  default=False,
                ),
                io.Custom(io_type="EASY_SAM3_EXTRA_CONFIG").Input(
                    "extra_config",
                    display_name="SAM3 Model Config",
                    tooltip="Extra configuration for the SAM3 model",
                    optional=True,
                ),
                io.String.Input(
                  "positive_coords",
                  display_name="positive_coords",
                  tooltip="Positive click coordinates as JSON: '[{\"x\": 50, \"y\": 120}]'",
                  optional=True,
                  force_input=True,
                ),
                io.String.Input(
                    "negative_coords",
                    display_name="negative_coords",
                    tooltip="Negative click coordinates as JSON: '[{\"x\": 150, \"y\": 300}]'",
                    optional=True,
                    force_input=True,
                ),
                io.BBOX.Input(
                    "bbox",
                    display_name="bbox",
                    optional=True,
                    tooltip="Bounding box as (x_min, y_min, x_max, y_max) or (x, y, width, height) tuple. Compatible with KJNodes Points Editor bbox output."
                ),
            ],
            outputs=[
                io.Mask.Output(
                    "output_masks",
                    display_name="masks",
                    tooltip="Tracked segmentation masks for all frames",
                ),
                io.String.Output(
                    "session_id_output",
                    display_name="session_id",
                ),
                io.Custom(io_type="EASY_SAM3_OBJECTS_OUTPUT").Output(
                    "objects",
                    display_name="objects"
                ),
                io.Mask.Output(
                    "obj_masks",
                    display_name="obj_masks"
                )
            ]
        )


    @classmethod
    def execute(cls, sam3_model, video_frames, prompt, frame_index, object_id, score_threshold_detection, new_det_thresh, propagation_direction, start_frame_index=0, max_frames_to_track=-1, close_after_propagation=True,  keep_model_loaded=False, session_id=None, extra_config=None, positive_coords=None, negative_coords=None,
                 bbox=None,) -> io.NodeOutput:
        offload_device = mm.unet_offload_device()

        video_predictor = sam3_model.get("model", None)
        device = sam3_model.get("device", torch.device("cpu"))
        dtype = sam3_model.get("dtype", torch.float32)
        segmentor = sam3_model.get("segmentor", None)
        B, H, W, _ = video_frames.shape

        if video_predictor is None or segmentor != "video":
            raise ValueError("Invalid SAM3 model. Please load a SAM3 model in 'video' mode")

        if frame_index > B - 1:
            logger.info(f"Frame index {frame_index} is out of bounds, setting to last frame {B - 1}")
            frame_index = B - 1

        # Set video model config
        video_predictor.model.score_threshold_detection = score_threshold_detection
        video_predictor.model.new_det_thresh = new_det_thresh
        
        # Set default values for video model parameters
        video_predictor.model.assoc_iou_thresh = 0.1
        video_predictor.model.det_nms_thresh = 0.1
        video_predictor.model.hotstart_delay = 15
        video_predictor.model.hotstart_unmatch_thresh = 8
        video_predictor.model.hotstart_dup_thresh = 8
        video_predictor.model.suppress_unmatched_only_within_hotstart = True
        video_predictor.model.min_trk_keep_alive = -1
        video_predictor.model.max_trk_keep_alive = 30
        video_predictor.model.init_trk_keep_alive = 30
        video_predictor.model.suppress_overlapping_based_on_recent_occlusion_threshold = 0.7
        video_predictor.model.suppress_det_close_to_boundary = False
        video_predictor.model.fill_hole_area = 16
        video_predictor.model.recondition_every_nth_frame = 16
        video_predictor.model.masklet_confirmation_enable = False
        video_predictor.model.decrease_trk_keep_alive_for_empty_masklets = False
        video_predictor.model.image_size = 1008
        
        # Override with extra_config if provided
        if extra_config is not None and isinstance(extra_config, dict):
            logger.info(f"Applying extra config: {extra_config}")
            for key, value in extra_config.items():
                if hasattr(video_predictor.model, key):
                    setattr(video_predictor.model, key, value)
                    logger.debug(f"Set {key} = {value}")
                else:
                    logger.warning(f"Model does not have attribute: {key}")

        # Start session
        video_pil = tensor_to_pil(video_frames)
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_pil,
                session_id=session_id
            )
        )

        session_id = response.get("session_id", None)
        if session_id is None:
            raise ValueError("Failed to start video prediction session")

        # Switch model to main device
        video_predictor.model.to(device)

        autocast_condition = not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():

            # Parse inputs with bounds checking
            pos_points, pos_count, pos_errors = parse_points(positive_coords, video_frames.shape)
            neg_points, neg_count, neg_errors = parse_points(negative_coords, video_frames.shape)
            # Combine points for refinement
            points = None
            point_labels = None
            if pos_points is not None and neg_points is not None:
                points = pos_points + neg_points
                point_labels = [1] * pos_count + [0] * neg_count
            elif pos_points is not None:
                points = pos_points
                point_labels = [1] * pos_count
            elif neg_points is not None:
                points = neg_points
                point_labels = [0] * neg_count

            # bbox (has bugs)
            bounding_boxes = None
            bounding_box_labels = None
            if bbox is not None:
                bbox_coords, bbox_count = parse_bbox(bbox, video_frames.shape)
                if bbox_coords is not None:
                    bounding_boxes = bbox_coords
                    bounding_box_labels = [1] * bbox_count

            # Add Prompt
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=prompt if prompt else None,
                    bounding_boxes=bounding_boxes,
                    bounding_box_labels=bounding_box_labels,
                    points=points,
                    point_labels=point_labels,
                    obj_id=object_id
                )
            )

            # Start to propagate
            # Output Masks
            output_masks = torch.zeros((B, H, W), dtype=torch.float32)

            # Initialize progress bar
            pbar = comfy.utils.ProgressBar(B)
            processed_frames = 0

            object_outputs = {
                "obj_ids":None,
                "obj_masks":None
            }
            object_masks = []

            for response in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=propagation_direction,
                    start_frame_index=start_frame_index,
                    max_frame_num_to_track=max_frames_to_track if max_frames_to_track != -1 else None,
                )
            ):
                frame_idx = response.get("frame_index", 0)
                outputs = response.get("outputs", {})
                obj_ids = outputs.get("out_obj_ids", None)
                if obj_ids is not None:
                    object_outputs["obj_ids"] = obj_ids
                if outputs:
                    if "out_binary_masks" in outputs:
                        mask = outputs["out_binary_masks"]
                        object_outputs["obj_masks"] = mask
                        if mask.shape[0] > 0:
                            # Convert mask to tensor and append to object_masks list
                            mask_tensor = torch.from_numpy(mask).float()
                            object_masks.append(mask_tensor)
                            
                            merged_mask = np.any(mask, axis=0).astype(np.float32)
                            frame_masks = torch.from_numpy(merged_mask)
                            output_masks[frame_idx] = frame_masks
                        else:
                            object_masks.append(torch.zeros((1, H, W)))
                    else:
                        object_masks.append(torch.zeros((1, H, W)))

                # Update progress bar
                processed_frames += 1
                pbar.update_absolute(processed_frames, B)

            # Switch model back to offload device
            if not keep_model_loaded:
                video_predictor.model.to(offload_device)
                mm.soft_empty_cache()

            # close session
            if close_after_propagation:
                video_predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )

        # Pad object_masks to have the same number of objects across all frames
        if len(object_masks) > 0:
            # Find the maximum number of objects across all frames
            max_num_objects = max(mask.shape[0] for mask in object_masks)
            
            # Pad each frame's masks to have max_num_objects
            padded_masks = []
            for mask in object_masks:
                num_objects = mask.shape[0]
                if num_objects < max_num_objects:
                    # Pad with zero masks
                    padding = torch.zeros((max_num_objects - num_objects, H, W))
                    padded_mask = torch.cat([mask, padding], dim=0)
                    padded_masks.append(padded_mask)
                else:
                    padded_masks.append(mask)
            
            object_masks = torch.stack(padded_masks, dim=0)
        else:
            # No masks detected, create empty tensor
            object_masks = torch.zeros((B, 1, H, W))

        return io.NodeOutput(output_masks, session_id, object_outputs, object_masks)


class Sam3VideoModelExtraConfig(io.ComfyNode):
    """Configure SAM3 video model parameters for fine-tuned control."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy sam3VideoModelExtraConfig",
            display_name="SAM3 Video Model Extra Config",
            category="EasyUse/Sam3",
            description="Configure advanced parameters for SAM3 video segmentation model",
            inputs=[
                io.Float.Input(
                    "assoc_iou_thresh",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="IoU threshold for detection-to-track matching"
                ),
                io.Float.Input(
                    "det_nms_thresh",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="IoU threshold for detection NMS (Non-Maximum Suppression)"
                ),
                io.Float.Input(
                    "new_det_thresh",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Threshold for a detection to be added as a new object"
                ),
                io.Int.Input(
                    "hotstart_delay",
                    default=15,
                    min=0,
                    max=100,
                    tooltip="Hold off outputs for N frames to remove unmatched/duplicate tracklets"
                ),
                io.Int.Input(
                    "hotstart_unmatch_thresh",
                    default=8,
                    min=0,
                    max=100,
                    tooltip="Remove tracklets unmatched for this many frames during hotstart"
                ),
                io.Int.Input(
                    "hotstart_dup_thresh",
                    default=8,
                    min=0,
                    max=100,
                    tooltip="Remove overlapping tracklets during hotstart"
                ),
                io.Boolean.Input(
                    "suppress_unmatched_within_hotstart",
                    default=True,
                    tooltip="If True, only suppress unmatched masks within hotstart period"
                ),
                io.Int.Input(
                    "min_trk_keep_alive",
                    default=-1,
                    min=-100,
                    max=0,
                    tooltip="Minimum keep-alive value (negative means immediate removal)"
                ),
                io.Int.Input(
                    "max_trk_keep_alive",
                    default=30,
                    min=0,
                    max=100,
                    tooltip="Maximum frames to keep a track alive without detections"
                ),
                io.Int.Input(
                    "init_trk_keep_alive",
                    default=30,
                    min=-10,
                    max=100,
                    tooltip="Initial keep-alive value when a new track is created"
                ),
                io.Float.Input(
                    "suppress_overlap_occlusion_thresh",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Threshold for suppressing overlapping objects based on recent occlusion (0.0 to disable)"
                ),
                io.Boolean.Input(
                    "suppress_det_at_boundary",
                    default=False,
                    tooltip="Suppress detections close to image boundaries"
                ),
                io.Int.Input(
                    "fill_hole_area",
                    default=16,
                    min=0,
                    max=1000,
                    tooltip="Fill holes in masks smaller than this area (in pixels)"
                ),
                io.Int.Input(
                    "recondition_every_nth_frame",
                    default=16,
                    min=-1,
                    max=1000,
                    tooltip="Recondition tracking every N frames (-1 to disable)"
                ),
                io.Boolean.Input(
                    "enable_masklet_confirmation",
                    default=False,
                    tooltip="Enable masklet confirmation to suppress unconfirmed tracklets"
                ),
                io.Boolean.Input(
                    "decrease_alive_for_empty_masks",
                    default=False,
                    tooltip="Decrease keep-alive counter for empty masklets (no valid masks)"
                ),
                io.Int.Input(
                    "image_size",
                    default=1008,
                    min=256,
                    max=2048,
                    step=8,
                    tooltip="Input image size for the model"
                ),
            ],
            outputs=[
                io.Custom(io_type="EASY_SAM3_EXTRA_CONFIG").Output(
                    display_name="extra_config",
                    tooltip="SAM3 model configuration dictionary"
                )
            ]
        )

    @classmethod
    def execute(
        cls,
        assoc_iou_thresh,
        det_nms_thresh,
        new_det_thresh,
        hotstart_delay,
        hotstart_unmatch_thresh,
        hotstart_dup_thresh,
        suppress_unmatched_within_hotstart,
        min_trk_keep_alive,
        max_trk_keep_alive,
        init_trk_keep_alive,
        suppress_overlap_occlusion_thresh,
        suppress_det_at_boundary,
        fill_hole_area,
        recondition_every_nth_frame,
        enable_masklet_confirmation,
        decrease_alive_for_empty_masks,
        image_size,
    ) -> io.NodeOutput:
        """Create a configuration dictionary for SAM3 model parameters."""
        
        config = {
            "assoc_iou_thresh": assoc_iou_thresh,
            "det_nms_thresh": det_nms_thresh,
            "new_det_thresh": new_det_thresh,
            "hotstart_delay": hotstart_delay,
            "hotstart_unmatch_thresh": hotstart_unmatch_thresh,
            "hotstart_dup_thresh": hotstart_dup_thresh,
            "suppress_unmatched_only_within_hotstart": suppress_unmatched_within_hotstart,
            "min_trk_keep_alive": min_trk_keep_alive,
            "max_trk_keep_alive": max_trk_keep_alive,
            "init_trk_keep_alive": init_trk_keep_alive,
            "suppress_overlapping_based_on_recent_occlusion_threshold": suppress_overlap_occlusion_thresh,
            "suppress_det_close_to_boundary": suppress_det_at_boundary,
            "fill_hole_area": fill_hole_area,
            "recondition_every_nth_frame": recondition_every_nth_frame,
            "masklet_confirmation_enable": enable_masklet_confirmation,
            "decrease_trk_keep_alive_for_empty_masklets": decrease_alive_for_empty_masks,
            "image_size": image_size,
        }
        
        logger.info(f"Created SAM3 model config with {len(config)} parameters")
        
        return io.NodeOutput(config)


class Sam3Visualization(io.ComfyNode):
    """Visualize segmentation masks with bounding boxes and scores on images."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy sam3Visualization",
            display_name="SAM3 Visualization",
            category="EasyUse/Sam3",
            description="Display mask visualization with objects, bounding boxes and confidence scores",
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Input image to visualize masks on"
                ),
                io.Mask.Input(
                    "obj_masks",
                    display_name="obj_masks",
                    tooltip="Individual object masks from Sam3 Image Segmentation node",
                ),
                io.Float.Input(
                    "scores",
                    display_name="scores",
                    min=0,
                    max=1,
                    step=0.0001,
                    tooltip="Confidence scores from Sam3 Image Segmentation node",
                    force_input=True,
                    optional=True,
                ),
                io.Float.Input(
                    "alpha",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Transparency of mask overlay (0=transparent, 1=opaque)"
                ),
                io.Int.Input(
                    "stroke_width",
                    default=5,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Width of the mask border stroke"
                ),
                io.Int.Input(
                    "font_size",
                    default=24,
                    min=8,
                    max=100,
                    step=1,
                    tooltip="Font size for confidence score text"
                )
            ],
            outputs=[
                io.Image.Output(
                    "visualization",
                    display_name="visualization",
                )
            ],
        )

    @classmethod
    def execute(cls, image, obj_masks, alpha=0.5, stroke_width=5, font_size=24, scores=None) -> io.NodeOutput:
        """
        Execute visualization of masks on images.
        
        Args:
            image: Input images tensor [B, H, W, C]
            boxes: List of bounding box tensors, one per image [N, 4] in [x0, y0, x1, y1] format
            scores: List of score tensors, one per image [N]
            masks: Optional list of mask tensors, one per image [N, H, W]
            alpha: Transparency for mask overlay
            
        Returns:
            Visualized images with masks, boxes and scores overlaid
        """
        B = image.shape[0]

        # Convert images to PIL format
        pil_images = tensor_to_pil(image)

        # Process each image
        visualized_images = []
        
        for idx in range(B):
            pil_image = pil_images[idx]
            raw_masks = obj_masks[idx] if obj_masks is not None else None
            # Create visualization
            # Note: If masks are None, visualize_masks_on_image will still draw boxes and scores
            vis_image = draw_visualize_image(
                pil_image,
                raw_masks,
                scores,
                None,
                alpha=alpha,
                stroke_width=stroke_width,
                font_size=font_size
            )

            # Convert back to tensor
            vis_tensor = pil_to_tensor(vis_image)
            visualized_images.append(vis_tensor)

        # Stack all visualized images
        output_images = torch.cat(visualized_images, dim=0)
        
        logger.info(f"Visualized {B} image(s) with masks, boxes and scores")
        
        # Return with preview UI
        return io.NodeOutput(output_images,)


class StringToBBox(io.ComfyNode):
    """Convert string coordinates to BBOX type."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy stringToBBox",
            display_name="String to BBox",
            category="EasyUse/Sam3",
            description="Convert x1,y1,x2,y2 format string to BBOX type",
            inputs=[
                io.String.Input(
                    "bbox_string",
                    default="",
                    multiline=True,
                    tooltip="Bounding box coordinates in format: x1,y1,x2,y2 (one per line for multiple boxes)",
                    force_input=True,
                ),
            ],
            outputs=[
                io.BBOX.Output(
                    "bbox",
                    display_name="bbox",
                    tooltip="Parsed bounding box in BBOX format"
                )
            ]
        )

    @classmethod
    def execute(cls, bbox_string) -> io.NodeOutput:
        """
        Convert string format bounding boxes to BBOX type.
        
        Args:
            bbox_string: String containing bbox coordinates in format "x1,y1,x2,y2"
                        Multiple boxes can be separated by newlines
            
        Returns:
            List of bounding boxes in format [{'startX': x1, 'startY': y1, 'endX': x2, 'endY': y2}, ...]
        """
        if not bbox_string or not bbox_string.strip():
            raise ValueError("Bounding box string cannot be empty")
        
        try:
            # Split by newlines for multiple boxes
            lines = [line.strip() for line in bbox_string.strip().split('\n') if line.strip()]
            
            bboxes = []
            for idx, line in enumerate(lines):
                # Split by comma
                parts = [p.strip() for p in line.split(',')]
                
                if len(parts) != 4:
                    raise ValueError(f"Line {idx + 1}: Expected 4 values (x1,y1,x2,y2), got {len(parts)}")
                
                try:
                    x1, y1, x2, y2 = [float(p) for p in parts]
                except ValueError as e:
                    raise ValueError(f"Line {idx + 1}: Could not convert coordinates to numbers: {e}")
                
                # Validate coordinates
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    raise ValueError(f"Line {idx + 1}: Coordinates must be non-negative, got ({x1}, {y1}, {x2}, {y2})")
                
                if x1 >= x2:
                    raise ValueError(f"Line {idx + 1}: x1 ({x1}) must be less than x2 ({x2})")
                
                if y1 >= y2:
                    raise ValueError(f"Line {idx + 1}: y1 ({y1}) must be less than y2 ({y2})")
                
                # Create bbox in KJNodes format
                bbox_dict = {
                    'startX': x1,
                    'startY': y1,
                    'endX': x2,
                    'endY': y2
                }
                bboxes.append(bbox_dict)
            
            logger.info(f"Parsed {len(bboxes)} bounding box(es) from string")
            
            return io.NodeOutput(bboxes)
            
        except Exception as e:
            raise ValueError(f"Error parsing bounding box string: {str(e)}")



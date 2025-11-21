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
from comfy_api.latest import ComfyExtension, io
from .sam3.logger import get_logger
from .utils import tensor_to_pil, pil_to_tensor, masks_to_tensor, join_image_with_alpha

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
                    default=0.60,
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
                )
                # io.String.Input(
                #     "coordinates_positive",
                #     display_name="coordinates_positive",
                #     optional=True,
                #     force_input=True,
                # ),
                # io.String.Input(
                #     "coordinates_negative",
                #     display_name="coordinates_negative",
                #     optional=True,
                #     force_input=True,
                # ),
                # io.BBOX.Input(
                #     "bboxes",
                #     display_name="bboxes",
                #     optional=True,
                # ),
                # io.Mask.Input(
                #     "mask",
                #     display_name="mask",
                #     optional=True,
                # ),
                # io.Boolean.Input(
                #     "enable_visualize",
                #     default=False,
                # ),
            ],
            outputs=[
                io.Mask.Output(
                    "output_masks",
                    display_name="masks",
                    is_output_list=True,
                    tooltip="Segmentation masks"
                ),
                io.Image.Output(
                    "output_images",
                    display_name="images",
                    is_output_list=True,
                    tooltip="Segmentation images",
                ),
                # io.Image.Output(
                #     "visualization",
                #     display_name="visualization",
                #     is_output_list=True,
                #     tooltip="When enable_visualize is True, the visualized image is output, otherwise the original image is output.",
                # )
            ]
        )

    @classmethod
    def execute(cls, sam3_model, images, prompt, threshold=0.3, keep_model_loaded=False, add_background='none', enable_visualize=False,  coordinates_positive=None, coordinates_negative=None, bboxes=None, mask=None) -> io.NodeOutput:
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

        # Todo: support for points and bbox prompts
        
        # # handle point coordinates
        # if coordinates_positive is not None:
        #     try:
        #         coordinates_positive = json.loads(coordinates_positive.replace("'", '"'))
        #         coordinates_positive = [(coord['x'], coord['y']) for coord in coordinates_positive]
        #         if coordinates_negative is not None:
        #             coordinates_negative = json.loads(coordinates_negative.replace("'", '"'))
        #             coordinates_negative = [(coord['x'], coord['y']) for coord in coordinates_negative]
        #     except:
        #         pass

        #     positive_point_coords = np.atleast_2d(np.array(coordinates_positive))

        #     if coordinates_negative is not None:
        #         negative_point_coords = np.array(coordinates_negative)
        #         # Ensure both positive and negative coords are lists of 2D arrays if individual_objects is True
        #         final_coords = np.concatenate((positive_point_coords, negative_point_coords), axis=0)
        #     else:
        #         final_coords = positive_point_coords

        # # Handle possible bboxes
        # if bboxes is not None:
        #     boxes_np_batch = []
        #     for bbox_list in bboxes:
        #         boxes_np = []
        #         for bbox in bbox_list:
        #             boxes_np.append(bbox)
        #         boxes_np = np.array(boxes_np)
        #         boxes_np_batch.append(boxes_np)
        #     final_box = np.array(boxes_np)
        #     final_labels = None

        # # handle labels
        # if coordinates_positive is not None:
        #     positive_point_labels = np.ones(len(positive_point_coords))

        #     if coordinates_negative is not None:
        #         negative_point_labels = np.zeros(len(negative_point_coords))  # 0 = negative
        #         final_labels = np.concatenate((positive_point_labels, negative_point_labels), axis=0)
        #     else:
        #         final_labels = positive_point_labels
        #     print("combined labels: ", final_labels)
        #     print("combined labels shape: ", final_labels.shape)

        # mask_list = []

        # Switch model to main device
        model.to(device)

        autocast_condition = not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
             # Convert tensor images to PIL format for processor
            pil_images = tensor_to_pil(images)
            num_frames = len(pil_images)
            logger.info(f"Processing {num_frames} image(s)")

            # Process each image separately to maintain correspondence
            output_masks = []
            output_images = []

            # Initialize progress bar

            pbar = comfy.utils.ProgressBar(num_frames)
            processed_frames = 0

            for idx, pil_img in enumerate(pil_images):
                # Set single image in processor
                state = processor.set_image(pil_img)

                # Prompt the model with text
                prompt_text = prompt.strip()
                if prompt_text:
                    state = processor.set_text_prompt(prompt_text, state)

                # TODO: Add support for points and bbox prompts

                # Get the masks and scores for this image
                masks = state.get('masks', None)
                boxes = state.get('boxes', None)
                scores = state.get('scores', None)

                # Handle empty results for this image
                if masks is None or len(masks) == 0:
                    logger.warning(f"No masks detected for image {idx}, using empty mask")
                    combined_mask = torch.zeros(H, W)
                else:
                    # Sort by scores (highest confidence first)
                    if scores is not None and len(scores) > 0:
                        logger.info(f"Image {idx}: detected {len(masks)} mask(s) with top score: {scores.max().item():.3f}")
                        top_indices = torch.argsort(scores, descending=True)
                        masks = masks[top_indices]
                        scores = scores[top_indices]

                    # Convert masks to tensor format
                    masks_tensor = masks_to_tensor(masks)

                    if masks_tensor is None or len(masks_tensor) == 0:
                        logger.warning(f"Failed to convert masks for image {idx}, using empty mask")
                        combined_mask = torch.zeros(H, W)
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
                    output_images.append([composited.squeeze(0)])
                else:
                    output_images.append([rgba_image.squeeze(0)])

                # Visualization: overlay mask on original image with color
                # if enable_visualize:
                #     vis_image = visualize_masks_on_image(
                #         pil_img,
                #         combined_mask,
                #         boxes,
                #         scores,
                #         alpha=0.5
                #     )
                #     vis_tensor = pil_to_tensor(vis_image)
                #     output_visualizations.append(vis_tensor)

                # Update progress bar
                processed_frames += 1
                pbar.update_absolute(processed_frames, num_frames)

            output_masks = torch.stack(output_masks, dim=0)
            logger.debug(f"Output masks shape: {output_masks.shape} (matches input images: {B})")

            # Clean up if not keeping model loaded
            if not keep_model_loaded:
                model.to(offload_device)
                mm.soft_empty_cache()

        return io.NodeOutput(output_masks, output_images)


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
                # io.String.Input(
                #     "coordinates_positive",
                #     display_name="coordinates_positive",
                #     optional=True,
                #     force_input=True,
                # ),
                # io.String.Input(
                #     "coordinates_negative",
                #     display_name="coordinates_negative",
                #     optional=True,
                #     force_input=True,
                # ),
                # io.BBOX.Input(
                #     "bboxes",
                #     display_name="bboxes",
                #     optional=True,
                # ),
                # io.Mask.Input(
                #     "mask",
                #     display_name="mask",
                #     optional=True,
                # )
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
                )
            ]
        )


    @classmethod
    def execute(cls, sam3_model, video_frames, prompt, frame_index, score_threshold_detection, new_det_thresh, propagation_direction, start_frame_index=0,close_after_propagation=True,  keep_model_loaded=False, session_id=None, extra_config=None,coordinates_positive=None, coordinates_negative=None,
                 bboxes=None, mask=None) -> io.NodeOutput:
        offload_device = mm.unet_offload_device()

        video_predictor = sam3_model.get("model", None)
        device = sam3_model.get("device", torch.device("cpu"))
        dtype = sam3_model.get("dtype", torch.float32)
        segmentor = sam3_model.get("segmentor", None)
        B, H, W, _ = video_frames.shape

        if video_predictor is None or segmentor != "video":
            raise ValueError("Invalid SAM3 model. Please load a SAM3 model in 'video' mode")

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
            # Add Prompt
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_index,
                    text=prompt,
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
            for response in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction=propagation_direction,
                    start_frame_index=start_frame_index,
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
                            # 合并的
                            merged_mask = np.any(mask, axis=0).astype(np.float32)
                            frame_masks = torch.from_numpy(merged_mask)
                            output_masks[frame_idx] = frame_masks

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

        return io.NodeOutput(output_masks, session_id, object_outputs)


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



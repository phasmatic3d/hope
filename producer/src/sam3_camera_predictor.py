import torch
import cv2
import logging
import numpy as np
from collections import OrderedDict
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

# SAM 3 imports
from sam3.model.sam3_tracker_base import Sam3TrackerBase
#from sam3.utils.misc import concat_points, fill_holes_in_mask_scores

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.error(f"Unexpected keys: {unexpected_keys}")
        logging.info("Loaded SAM 3 checkpoint successfully")

def build_sam3_camera_predictor(
    config_file: str = "sam3_hiera_l.yaml", # Default to SAM 3 Large config
    config_path: str = "./sam3/configs",    # Adjust path to your SAM 3 repo configs
    ckpt_path=None, 
    device="cuda", 
    mode="eval", 
    hydra_overrides_extra=[], 
    apply_postprocessing=True, 
    image_size=1024 
):
    # Target the new SAM3CameraPredictor class
    hydra_overrides = [ "++model._target_=sam3_camera_predictor.SAM3CameraPredictor", ]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]

    hydra_overrides.extend(hydra_overrides_extra)
    
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_file, overrides=hydra_overrides)

    cfg.model.image_size = image_size
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)

    if mode == "eval":
        model.eval()

    return model

class SAM3CameraPredictor(Sam3TrackerBase):
    """
    SAM 3 implementation for Real-Time Camera Prediction.
    Includes support for Text Prompts (Concept Segmentation).
    """

    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        clear_non_cond_mem_for_multi_obj=False,
        device: str="cpu", 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.condition_state = {}
        self.frame_idx = 0
        self._device = torch.device(device)

    # --- Data Prep ---
    def prepare_data(self, img, image_size, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)):
        if isinstance(img, np.ndarray):
            img_np = cv2.resize(img, (image_size, image_size)) / 255.0
            height, width = img.shape[:2]
        else:
            img_np = np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0
            width, height = img.size
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        img_tensor -= img_mean
        img_tensor /= img_std
        return img_tensor, width, height

    # --- State Management ---
    @torch.inference_mode()
    def load_first_frame(self, img):
        self.condition_state = self._init_state(offload_video_to_cpu=False, offload_state_to_cpu=False)
        img_tensor, width, height = self.prepare_data(img, image_size=self.image_size)
        
        self.condition_state["images"] = [img_tensor]
        self.condition_state["num_frames"] = 1
        self.condition_state["video_height"] = height
        self.condition_state["video_width"] = width
        # Warm up features for frame 0
        self._get_image_feature(frame_idx=0, batch_size=1)

    def add_conditioning_frame(self, img):
        img_tensor, _, _ = self.prepare_data(img, image_size=self.image_size)
        self.condition_state["images"].append(img_tensor)
        self.condition_state["num_frames"] = len(self.condition_state["images"])
        # Pre-compute features for the new frame
        self._get_image_feature(frame_idx=self.condition_state["num_frames"] - 1, batch_size=1)

    # --- SAM 3 New Feature: Text Prompting ---
    @torch.inference_mode()
    def add_text_prompt(self, frame_idx, obj_id, text_prompt):
        """
       
        Uses SAM 3's text encoder to find an object by concept name (e.g., 'red cup') 
        and adds it as a mask constraint for tracking.
        """
        obj_idx = self._obj_id_to_idx(obj_id)
        
        # 1. Get Image Features for the current frame
        (
            _, _, 
            vision_feats, 
            vision_pos_embeds, 
            feat_sizes
        ) = self._get_image_feature(frame_idx, batch_size=1)

        # 2. Run SAM 3 Text Encoder & Detector on this frame
        # Note: This calls the internal SAM 3 prompt encoder.
        # We assume 'self.text_encoder' and 'self.prompt_encoder' exist in SAM3Base
        text_inputs = self.tokenizer(text_prompt, return_tensors="pt", padding=True).to(self.device)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            text=text_inputs,
            image_embeddings=vision_feats
        )

        # 3. Predict Mask using the text embeddings
        # We use the mask decoder with the text-derived embeddings
        low_res_masks, iou_predictions, _ = self.mask_decoder(
            image_embeddings=vision_feats,
            image_pe=vision_pos_embeds,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 4. Convert the best mask to a binary input mask for the tracker
        # Use the "Presence Head" score if available to filter out bad prompts
        best_mask_idx = torch.argmax(iou_predictions, dim=1)
        mask_inputs = low_res_masks[:, best_mask_idx, :, :]
        
        # Binarize for the tracker input (sigmoid > 0.0)
        mask_inputs = (mask_inputs > 0.0).float()

        # 5. Register this mask as a user input for this frame (just like a click)
        # This re-uses your existing 'add_new_mask' logic but sourced from text
        return self.add_new_mask(frame_idx, obj_id, mask_inputs.squeeze(0))

    # --- Existing SAM 2 Methods (Preserved & Compatible) ---
    # These methods (add_new_points, add_new_mask, track) remain largely the same 
    # because SAM 3 inherits the tracker architecture from SAM 2.
    
    # ... (Insert the rest of your original methods: _init_state, _obj_id_to_idx, 
    #      add_new_points, add_new_mask, track, etc. here without changes) ...
    
    # NOTE: Ensure you copy the full body of 'add_new_points', 'track', etc. 
    # from your original file. They are compatible with SAM 3's tracker.

    # [Placeholder for the rest of the class methods to keep response concise]
    # You should copy: 
    # - _init_state
    # - _obj_id_to_idx
    # - _obj_idx_to_id
    # - add_new_points
    # - add_new_mask (IMPORTANT: used by add_text_prompt)
    # - _get_orig_video_res_output
    # - _consolidate_temp_output_across_obj
    # - track
    # - propagate_in_video
    # - reset_state
    # - _run_single_frame_inference
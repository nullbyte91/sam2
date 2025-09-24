# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
from torch import nn
from typing import Any, Dict, Tuple
from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2
import os

# Constants
IMAGE_SIZE = 1024
DEFAULT_NUM_POINTS = 10

# Model configurations
MODEL_CONFIGS: Dict[str, str] = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
}

class SAM2Encoder(nn.Module):
    """Encoder module for SAM2 model that processes input images into feature maps."""
    
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed 

    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input image through the encoder.
        
        Args:
            input_image: Input image tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of three feature maps at different resolutions
        """
        backbone_out = self.image_encoder(input_image)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.num_feature_levels:]

        feature_map_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feature_maps = [feat.permute(1, 2, 0).reshape(input_image.shape[0], -1, *feat_size)
                 for feat, feat_size in zip(vision_feats[::-1], feature_map_sizes[::-1])][::-1]

        return feature_maps[0], feature_maps[1], feature_maps[2]

class SAM2Decoder(nn.Module):
    """Decoder module for SAM2 model that generates masks from image features and prompts."""
    
    def __init__(
            self,
            sam_model: SAM2Base,
            multimask_output: bool
    ) -> None:
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.multimask_output = multimask_output

    @torch.no_grad()
    def forward(
            self,
            image_embed: torch.Tensor,
            high_res_feats_0: torch.Tensor,
            high_res_feats_1: torch.Tensor,
            point_coords: torch.Tensor,
            point_labels: torch.Tensor,
            mask_input: torch.Tensor,
            has_mask_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate masks from image features and prompts.
        
        Args:
            image_embed: Image embeddings
            high_res_feats_0: High resolution features at first scale
            high_res_feats_1: High resolution features at second scale
            point_coords: Point coordinates for prompts
            point_labels: Point labels for prompts
            mask_input: Input mask prompts
            has_mask_input: Flag indicating if mask input is provided
            
        Returns:
            Tuple of predicted masks and IoU predictions
        """
        sparse_embedding = self._encode_point_prompts(point_coords, point_labels)
        dense_embedding = self._encode_mask_prompts(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]

        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=True,
            high_res_features=high_res_feats,
        )

        if self.multimask_output:
            masks = masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            masks, iou_predictions = self.mask_decoder._dynamic_multimask_via_stability(masks, iou_predictions)

        return masks, iou_predictions

    def _encode_point_prompts(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        """Encode point prompts into embeddings."""
        point_coords = point_coords + 0.5

        padding_point = torch.zeros((point_coords.shape[0], 1, 2), device=point_coords.device)
        padding_label = -torch.ones((point_labels.shape[0], 1), device=point_labels.device)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        # Normalize coordinates
        point_coords = point_coords / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
                point_labels == -1
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _encode_mask_prompts(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        """Encode mask prompts into embeddings."""
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
                1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export SAM2 model to ONNX format')
    parser.add_argument('model_type', type=str,
                      choices=list(MODEL_CONFIGS.keys()),
                      help='Type of SAM2 model to export')
    parser.add_argument('checkpoint', type=str,
                      help='Path to the model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Directory to save exported ONNX models (default: ./output)')
    return parser.parse_args()

def get_model_config(model_type: str) -> str:
    if model_type not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model type: {model_type}. Available types: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type]

def main() -> None:
    """Main function to export SAM2 model to ONNX format."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model_cfg = get_model_config(args.model_type)
    print(f"Loading model from config: {model_cfg}")
    print(f"Using checkpoint: {args.checkpoint}")
    
    sam2_model = build_sam2(model_cfg, args.checkpoint, device="cpu")
    
    # Export encoder
    print("\nExporting encoder...")
    img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).cpu()
    sam2_encoder = SAM2Encoder(sam2_model).cpu()
    
    torch.onnx.export(sam2_encoder,
                     img,
                     f"{args.output_dir}/{args.model_type}_encoder.onnx",
                     export_params=True,
                     opset_version=17,
                     do_constant_folding=True,
                     input_names=['image'],
                     output_names=['high_res_feats_0', 'high_res_feats_1', 'image_embed'],
                     dynamic_axes={
                         "image": {0: "batch_size"},
                         "high_res_feats_0": {0: "batch_size"},
                         "high_res_feats_1": {0: "batch_size"},
                         "image_embed": {0: "batch_size"},
                     })
    
    # Export decoder
    print("\nExporting decoder...")
    image_embed = torch.randn(1, 256, 64, 64).cpu()
    high_res_feats_0 = torch.randn(1, 32, 256, 256).cpu()
    high_res_feats_1 = torch.randn(1, 64, 128, 128).cpu()
    
    sam2_decoder = SAM2Decoder(sam2_model, multimask_output=False).cpu()
    
    embed_size = (sam2_model.image_size // sam2_model.backbone_stride, 
                 sam2_model.image_size // sam2_model.backbone_stride)
    mask_input_size = [4 * x for x in embed_size]
    
    point_coords = torch.randint(low=0, high=IMAGE_SIZE, size=(DEFAULT_NUM_POINTS, 2, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(DEFAULT_NUM_POINTS, 2), dtype=torch.float)
    mask_input = torch.randn(DEFAULT_NUM_POINTS, 1, *mask_input_size, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    
    torch.onnx.export(sam2_decoder,
                     (image_embed, high_res_feats_0, high_res_feats_1, 
                      point_coords, point_labels, mask_input, has_mask_input),
                     f"{args.output_dir}/{args.model_type}_decoder.onnx",
                     export_params=True,
                     opset_version=16,
                     do_constant_folding=True,
                     input_names=['image_embed', 'high_res_feats_0', 'high_res_feats_1', 
                                'point_coords', 'point_labels', 'mask_input', 'has_mask_input'],
                     output_names=['masks', 'iou_predictions'],
                     dynamic_axes={
                         "point_coords": {0: "num_labels"},
                         "point_labels": {0: "num_labels"},
                         "mask_input": {0: "num_labels"}
                     })
    
    print(f"\nExport completed! Models saved to {args.output_dir}/")

if __name__ == '__main__':
    main()

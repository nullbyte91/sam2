The export script converts only the image segmentation components of SAM2, splitting them into two separate ONNX models:
## SAM2Encoder (Image Processing Pipeline)
Components Included:
1. Hiera Backbone (image_encoder.trunk)
2. FPN Neck (image_encoder.neck)
3. High-res Feature Convolutions (conv_s0, conv_s1)
4. No-memory Embedding (no_mem_embed)
```bash
Input:  image [B, 3, 1024, 1024]
Output: high_res_feats_0 [B, 32, 256, 256]   # 1/4 resolution
        high_res_feats_1 [B, 64, 128, 128]   # 1/8 resolution  
        image_embed      [B, 256, 64, 64]    # 1/16 resolution
```

## SAM2Decoder (Mask Generation Pipeline)
Components Included:
1. Mask Decoder (sam_mask_decoder)
2. Prompt Encoder (prompt_encoder)
3. Point/Mask Prompt Processing
```bash
Input:  image_embed      [B, 256, 64, 64]
        high_res_feats_0 [B, 32, 256, 256]
        high_res_feats_1 [B, 64, 128, 128]
        point_coords     [N, 2, 2]           # Point coordinates
        point_labels     [N, 2]              # Point labels
        mask_input       [N, 1, H, W]        # Mask prompts
        has_mask_input   [1]                 # Mask flag

Output: masks           [B, 1, H, W]        # Predicted masks
        iou_predictions [B, 1]              # Quality scores
```

The below components are removed from the main archiecture:
Video/Temporal Components (Removed):
1. Memory Attention System (memory_attention)
2. Memory Encoder (memory_encoder)
3. Video Predictor (SAM2VideoPredictor)
4. Object Pointers (obj_ptrs)
5. Temporal Processing (cross-frame reasoning)
6. Memory Bank Management



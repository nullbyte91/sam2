#!/usr/bin/env python3
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

"""
SAM2 ONNX to TensorRT Conversion Script

This script converts SAM2 ONNX models to TensorRT engines for deployment
with NVIDIA Triton Inference Server.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Constants
DEFAULT_ONNX_DIR = "onnx_module"
DEFAULT_MODEL_REPO = "model_repository"

# Model configurations
MODEL_SIZES = ["tiny", "small", "base_plus", "large"]

@dataclass
class TensorRTConfig:
    """Configuration for TensorRT engine building."""
    fp16: bool = True
    workspace_size: str = "6144M"
    optimization_level: int = 5
    avg_timing: int = 10
    verbose: bool = True


@dataclass
class DecoderShapes:
    """Dynamic shape configurations for decoder model."""
    # Min shapes (batch_size=1)
    min_batch: int = 1
    # Opt shapes (batch_size=32)
    opt_batch: int = 32
    # Max shapes (batch_size=64)
    max_batch: int = 64
    
    # Fixed dimensions
    num_points: int = 2
    point_dim: int = 2
    mask_channels: int = 1
    mask_size: int = 256
    embed_channels: int = 256
    embed_h: int = 64
    embed_w: int = 64
    high_res_0_channels: int = 32
    high_res_0_size: int = 256
    high_res_1_channels: int = 64
    high_res_1_size: int = 128


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert SAM2 ONNX models to TensorRT engines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a specific model
  python convert_onnx_to_tensorrt.py --model-size large
  
  # Convert all models
  python convert_onnx_to_tensorrt.py --all
  
  # Use custom directories
  python convert_onnx_to_tensorrt.py --model-size small --onnx-dir ./onnx --output-dir ./engines
  
  # Use custom trtexec path
  python convert_onnx_to_tensorrt.py --model-size large --trtexec-path /usr/local/bin/trtexec
        """
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        choices=MODEL_SIZES,
        help='Size of SAM2 model to convert (tiny, small, base_plus, large)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all available models'
    )
    
    parser.add_argument(
        '--onnx-dir',
        type=str,
        default=DEFAULT_ONNX_DIR,
        help=f'Directory containing ONNX models (default: {DEFAULT_ONNX_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_MODEL_REPO,
        help=f'Output directory for TensorRT engines (default: {DEFAULT_MODEL_REPO})'
    )
    
    parser.add_argument(
        '--trtexec-path',
        type=str,
        default='trtexec',
        help='Path to trtexec binary (default: trtexec from PATH)'
    )
    
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 precision (use FP32)'
    )
    
    parser.add_argument(
        '--workspace-size',
        type=str,
        default='6144M',
        help='TensorRT workspace size (default: 6144M)'
    )
    
    parser.add_argument(
        '--optimization-level',
        type=int,
        default=5,
        choices=[0, 1, 2, 3, 4, 5],
        help='TensorRT builder optimization level (default: 5)'
    )
    
    parser.add_argument(
        '--encoder-only',
        action='store_true',
        help='Convert encoder only'
    )
    
    parser.add_argument(
        '--decoder-only',
        action='store_true',
        help='Convert decoder only'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.all and not args.model_size:
        parser.error("Either --model-size or --all must be specified")
    
    if args.all and args.model_size:
        parser.error("Cannot specify both --model-size and --all")
    
    if args.encoder_only and args.decoder_only:
        parser.error("Cannot specify both --encoder-only and --decoder-only")
    
    return args


def check_trtexec_available(trtexec_path: str) -> bool:
    """Check if trtexec is available."""
    try:
        result = subprocess.run(
            [trtexec_path, "--help"],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_onnx_models(onnx_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find available ONNX models in the directory.
    
    Returns:
        List of tuples (model_size, encoder_path, decoder_path)
    """
    onnx_path = Path(onnx_dir)
    if not onnx_path.exists():
        print(f"❌ ONNX directory not found: {onnx_dir}")
        return []
    
    models = []
    for model_size in MODEL_SIZES:
        # Try different naming patterns
        patterns = [
            (f"sam2.1_hiera_{model_size}_encoder.onnx", f"sam2.1_hiera_{model_size}_decoder.onnx"),
            (f"sam2_hiera_{model_size}_encoder.onnx", f"sam2_hiera_{model_size}_decoder.onnx"),
        ]
        
        for encoder_name, decoder_name in patterns:
            encoder_path = onnx_path / encoder_name
            decoder_path = onnx_path / decoder_name
            
            if encoder_path.exists() and decoder_path.exists():
                models.append((model_size, str(encoder_path), str(decoder_path)))
                break
    
    return models


def list_available_models(onnx_dir: str) -> None:
    """List all available ONNX models."""
    print("\n📋 Available ONNX models:")
    print("=" * 60)
    
    models = find_onnx_models(onnx_dir)
    if not models:
        print("  No ONNX models found!")
        return
    
    for model_size, encoder_path, decoder_path in models:
        encoder_size = Path(encoder_path).stat().st_size / (1024 * 1024)
        decoder_size = Path(decoder_path).stat().st_size / (1024 * 1024)
        print(f"  ✓ SAM2 Hiera {model_size.upper()}")
        print(f"    Encoder: {encoder_path} ({encoder_size:.1f} MB)")
        print(f"    Decoder: {decoder_path} ({decoder_size:.1f} MB)")
        print()


def build_encoder_engine(
    model_size: str,
    onnx_path: str,
    output_path: str,
    trtexec_path: str,
    config: TensorRTConfig
) -> bool:
    """Build TensorRT engine for encoder model."""
    print(f"\n🏗️  Building {model_size} encoder engine...")
    print("=" * 60)
    
    # Prepare trtexec command
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--shapes=image:1x3x1024x1024",
        f"--memPoolSize=workspace:{config.workspace_size}",
        f"--builderOptimizationLevel={config.optimization_level}",
        f"--avgTiming={config.avg_timing}",
    ]
    
    if config.fp16:
        cmd.append("--fp16")
    
    if config.verbose:
        cmd.append("--verbose")
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Encoder engine built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build encoder engine: {e}")
        return False


def build_decoder_engine(
    model_size: str,
    onnx_path: str,
    output_path: str,
    trtexec_path: str,
    config: TensorRTConfig,
    shapes: DecoderShapes
) -> bool:
    """Build TensorRT engine for decoder model."""
    print(f"\n🏗️  Building {model_size} decoder engine...")
    print("=" * 60)
    
    # Format shape strings
    min_shapes = (
        f"point_coords:{shapes.min_batch}x{shapes.num_points}x{shapes.point_dim},"
        f"point_labels:{shapes.min_batch}x{shapes.num_points},"
        f"mask_input:{shapes.min_batch}x{shapes.mask_channels}x{shapes.mask_size}x{shapes.mask_size},"
        f"image_embed:1x{shapes.embed_channels}x{shapes.embed_h}x{shapes.embed_w},"
        f"high_res_feats_0:1x{shapes.high_res_0_channels}x{shapes.high_res_0_size}x{shapes.high_res_0_size},"
        f"high_res_feats_1:1x{shapes.high_res_1_channels}x{shapes.high_res_1_size}x{shapes.high_res_1_size},"
        f"has_mask_input:1"
    )
    
    opt_shapes = (
        f"point_coords:{shapes.opt_batch}x{shapes.num_points}x{shapes.point_dim},"
        f"point_labels:{shapes.opt_batch}x{shapes.num_points},"
        f"mask_input:{shapes.opt_batch}x{shapes.mask_channels}x{shapes.mask_size}x{shapes.mask_size},"
        f"image_embed:1x{shapes.embed_channels}x{shapes.embed_h}x{shapes.embed_w},"
        f"high_res_feats_0:1x{shapes.high_res_0_channels}x{shapes.high_res_0_size}x{shapes.high_res_0_size},"
        f"high_res_feats_1:1x{shapes.high_res_1_channels}x{shapes.high_res_1_size}x{shapes.high_res_1_size},"
        f"has_mask_input:1"
    )
    
    max_shapes = (
        f"point_coords:{shapes.max_batch}x{shapes.num_points}x{shapes.point_dim},"
        f"point_labels:{shapes.max_batch}x{shapes.num_points},"
        f"mask_input:{shapes.max_batch}x{shapes.mask_channels}x{shapes.mask_size}x{shapes.mask_size},"
        f"image_embed:1x{shapes.embed_channels}x{shapes.embed_h}x{shapes.embed_w},"
        f"high_res_feats_0:1x{shapes.high_res_0_channels}x{shapes.high_res_0_size}x{shapes.high_res_0_size},"
        f"high_res_feats_1:1x{shapes.high_res_1_channels}x{shapes.high_res_1_size}x{shapes.high_res_1_size},"
        f"has_mask_input:1"
    )
    
    # Prepare trtexec command
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
        f"--memPoolSize=workspace:{config.workspace_size}",
        f"--builderOptimizationLevel={config.optimization_level}",
        f"--avgTiming={config.avg_timing}",
    ]
    
    if config.fp16:
        cmd.append("--fp16")
    
    if config.verbose:
        cmd.append("--verbose")
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Decoder engine built successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build decoder engine: {e}")
        return False


def convert_model(
    model_size: str,
    encoder_onnx: str,
    decoder_onnx: str,
    output_dir: str,
    trtexec_path: str,
    config: TensorRTConfig,
    encoder_only: bool = False,
    decoder_only: bool = False
) -> bool:
    """Convert a single model from ONNX to TensorRT."""
    print(f"\n{'='*60}")
    print(f"🔨 Converting SAM2 Hiera {model_size.upper()}")
    print(f"{'='*60}")
    
    # Create output directories
    encoder_dir = Path(output_dir) / f"sam2_{model_size}_encoder" / "1"
    decoder_dir = Path(output_dir) / f"sam2_{model_size}_decoder" / "1"
    
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)
    
    encoder_engine = encoder_dir / "model.engine"
    decoder_engine = decoder_dir / "model.engine"
    
    # Use absolute paths
    encoder_onnx_abs = os.path.abspath(encoder_onnx)
    decoder_onnx_abs = os.path.abspath(decoder_onnx)
    encoder_engine_abs = os.path.abspath(encoder_engine)
    decoder_engine_abs = os.path.abspath(decoder_engine)
    
    success = True
    
    # Build encoder
    if not decoder_only:
        if not build_encoder_engine(
            model_size,
            encoder_onnx_abs,
            encoder_engine_abs,
            trtexec_path,
            config
        ):
            success = False
    
    # Build decoder
    if not encoder_only and success:
        decoder_shapes = DecoderShapes()
        if not build_decoder_engine(
            model_size,
            decoder_onnx_abs,
            decoder_engine_abs,
            trtexec_path,
            config,
            decoder_shapes
        ):
            success = False
    
    if success:
        print(f"\n✅ {model_size.upper()} conversion completed!")
        print(f"\n📊 Engine file sizes:")
        if not decoder_only and encoder_engine.exists():
            size_mb = encoder_engine.stat().st_size / (1024 * 1024)
            print(f"  Encoder: {encoder_engine} ({size_mb:.1f} MB)")
        if not encoder_only and decoder_engine.exists():
            size_mb = decoder_engine.stat().st_size / (1024 * 1024)
            print(f"  Decoder: {decoder_engine} ({size_mb:.1f} MB)")
    
    return success


def print_summary(converted_models: List[str], output_dir: str) -> None:
    """Print conversion summary and next steps."""
    print("\n" + "="*60)
    print("🎉 Conversion Summary")
    print("="*60)
    
    if not converted_models:
        print("❌ No models were successfully converted.")
        return
    
    print(f"\n✅ Successfully converted {len(converted_models)} model(s):")
    for model_size in converted_models:
        print(f"\n  SAM2 Hiera {model_size.upper()}:")
        print(f"    Encoder: {output_dir}/sam2_{model_size}_encoder/1/model.engine")
        print(f"    Decoder: {output_dir}/sam2_{model_size}_decoder/1/model.engine")
    
    print("\n" + "="*60)
    print("🚀 Next Steps")
    print("="*60)
    print("\n1. Update your Triton model configs to point to the new engines")
    print("\n2. Test with Triton server:")
    print(f"   docker run --gpus all --rm \\")
    print(f"     -v ./{output_dir}:/models \\")
    print(f"     -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    print(f"     nvcr.io/nvidia/tritonserver:25.01-py3 \\")
    print(f"     tritonserver --model-repository=/models")
    print("\n💡 Note: These engines are built with TensorRT 24.11")
    print("   which should be compatible with Triton 25.01")
    print()


def main() -> int:
    """Main function."""
    args = parse_args()
    
    print("🔧 SAM2 ONNX to TensorRT Conversion")
    print("="*60)
    
    # Check trtexec availability
    if not check_trtexec_available(args.trtexec_path):
        print(f"❌ trtexec is not available at: {args.trtexec_path}")
        print("\nPlease install TensorRT or provide the correct path using --trtexec-path")
        print("\nTo install TensorRT:")
        print("  - Download from: https://developer.nvidia.com/tensorrt")
        print("  - Or install via pip: pip install tensorrt")
        return 1
    
    print(f"✓ Found trtexec at: {args.trtexec_path}")
    
    # List available models
    list_available_models(args.onnx_dir)
    
    # Find models to convert
    available_models = find_onnx_models(args.onnx_dir)
    if not available_models:
        print("❌ No ONNX models found. Please export models first using export_sam2_onnx.py")
        return 1
    
    # Determine which models to convert
    if args.all:
        models_to_convert = available_models
    else:
        models_to_convert = [
            (size, enc, dec) for size, enc, dec in available_models
            if size == args.model_size
        ]
        
        if not models_to_convert:
            print(f"❌ Model size '{args.model_size}' not found in {args.onnx_dir}")
            return 1
    
    # Create TensorRT config
    trt_config = TensorRTConfig(
        fp16=not args.no_fp16,
        workspace_size=args.workspace_size,
        optimization_level=args.optimization_level,
        verbose=True
    )
    
    # Convert models
    converted_models = []
    for model_size, encoder_onnx, decoder_onnx in models_to_convert:
        if convert_model(
            model_size,
            encoder_onnx,
            decoder_onnx,
            args.output_dir,
            args.trtexec_path,
            trt_config,
            args.encoder_only,
            args.decoder_only
        ):
            converted_models.append(model_size)
    
    # Print summary
    print_summary(converted_models, args.output_dir)
    
    return 0 if converted_models else 1


if __name__ == '__main__':
    sys.exit(main())

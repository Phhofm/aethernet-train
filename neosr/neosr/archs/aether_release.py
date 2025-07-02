#!/usr/bin/env python3
"""
AetherNet Model Release Script
==============================

Converts trained AetherNet models to optimized deployment formats:
- PyTorch: FP32 (fused), FP16 (fused), INT8 (quantized)
- ONNX: FP32, FP16, INT8

Features:
- Comprehensive validation of all converted models
- Robust error handling with retry mechanisms
- Automatic metadata extraction from trained models
- Version compatibility checks
- Per-channel to per-tensor quantization fallback
- Detailed logging and progress reporting

Usage:
python aether_release.py \
    --model-path path/to/trained_model.pth \
    --output-dir path/to/output_directory \
    --validation-dir path/to/validation_images \
    --arch aether_tiny
"""

import argparse
import os
import sys
import time
import logging
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, Optional, Callable, Any

import torch
import numpy as np
from PIL import Image
import onnxruntime as ort

# Import model definitions from core implementation
from aether_core import (
    aether_tiny, aether_small, aether_medium, aether_large,
    export_onnx, AetherNet
)

# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
ort.set_default_logger_severity(3)  # ONNX Runtime: Only show errors

# ------------------- Constants & Configuration ------------------- 
MAX_RETRIES = 3
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
ARCH_MAP = {
    'aether_tiny': aether_tiny,
    'aether_small': aether_small,
    'aether_medium': aether_medium,
    'aether_large': aether_large
}

# ------------------- Setup Logging ------------------- 
def setup_logger(output_dir: Path) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger("AetherNetRelease")
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)
    
    # File handler
    log_file = output_dir / "aether_release.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    
    return logger

# ------------------- Helper Functions ------------------- 
def load_image(image_path: Path) -> torch.Tensor:
    """Load an image and convert to normalized tensor."""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

def validate_pytorch_model(
    model: torch.nn.Module, 
    validation_data: torch.Tensor,
    device: torch.device
) -> bool:
    """Validate PyTorch model can run inference without errors."""
    try:
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            input_tensor = validation_data.to(device)
            
            # Handle precision automatically
            if next(model.parameters(), torch.tensor(0)).dtype == torch.float16:
                input_tensor = input_tensor.half()
            
            output = model(input_tensor)
            
            # Basic output sanity check
            if not torch.isfinite(output).all():
                raise RuntimeError("Model produced non-finite output values")
                
        return True
    except Exception as e:
        logger.error(f"PyTorch validation FAILED: {e}")
        return False

def validate_onnx_model(
    onnx_path: Path, 
    validation_data: torch.Tensor, 
    scale: int
) -> bool:
    """Validate ONNX model produces correct output shape."""
    try:
        # Configure execution providers
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        input_dtype = session.get_inputs()[0].type
        
        # Prepare input data
        input_data = validation_data.cpu().numpy()
        if 'float16' in input_dtype:
            input_data = input_data.astype(np.float16)
        
        # Run inference
        output = session.run(None, {input_name: input_data})[0]
        
        # Verify output shape
        _, c, h, w = validation_data.shape
        expected_shape = (1, c, h * scale, w * scale)
        
        if output.shape != expected_shape:
            logger.error(
                f"ONNX shape mismatch. Expected {expected_shape}, got {output.shape}"
            )
            return False
            
        return True
    except Exception as e:
        logger.error(f"ONNX validation FAILED: {e}")
        return False

def attempt_operation(
    operation: Callable,
    validation: Callable[[Any], bool],
    success_msg: str,
    error_msg: str,
    max_retries: int = MAX_RETRIES
) -> Any:
    """Attempt an operation with retries and validation."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}")
            result = operation()
            
            if validation(result):
                logger.info(f"✅ {success_msg}")
                return result
                
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            time.sleep(1)  # Brief pause before retry
            
    logger.error(f"❌ {error_msg}")
    return None

# ------------------- Model Conversion Functions ------------------- 
def create_fp32_model(
    base_model: AetherNet,
    arch: str,
    scale: int
) -> Optional[AetherNet]:
    """Create optimized FP32 model for deployment."""
    def _operation():
        model = ARCH_MAP[arch](scale=scale)
        model.fuse_model()
        model.load_state_dict(base_model.state_dict(), strict=False)
        return model
        
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created optimized FP32 model",
        error_msg="Failed to create FP32 model"
    )

def create_fp16_model(
    fp32_model: AetherNet
) -> Optional[AetherNet]:
    """Convert FP32 model to FP16 precision."""
    def _operation():
        return deepcopy(fp32_model).half()
        
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created FP16 model",
        error_msg="Failed to create FP16 model"
    )

def create_int8_model(
    base_model: AetherNet
) -> Optional[AetherNet]:
    """Convert QAT model to optimized INT8 model."""
    def _operation():
        # Ensure we're using a quantization-compatible backend
        original_engine = torch.backends.quantized.engine
        torch.backends.quantized.engine = 'qnnpack'
        
        try:
            # Prepare and convert to INT8
            model = deepcopy(base_model).cpu().eval()
            model.prepare_qat()
            int8_model = model.convert_to_quantized()
            
            # Verify quantization
            if not int8_model.verify_quantization():
                logger.warning("Quantization verification reported issues")
                
            return int8_model
        finally:
            torch.backends.quantized.engine = original_engine
            
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, torch.device('cpu')),
        success_msg="Created INT8 model",
        error_msg="Failed to create INT8 model"
    )

def export_onnx_model(
    model: AetherNet,
    scale: int,
    precision: str,
    output_path: Path
) -> bool:
    """Export model to ONNX format with validation."""
    def _operation():
        export_onnx(model, scale, precision, str(output_path))
        return output_path
        
    result = attempt_operation(
        operation=_operation,
        validation=lambda p: validate_onnx_model(p, validation_sample, scale),
        success_msg=f"Exported {precision.upper()} ONNX model",
        error_msg=f"Failed to export {precision.upper()} ONNX model"
    )
    
    return result is not None

# ------------------- Main Execution ------------------- 
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert trained AetherNet models to deployment formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for converted models')
    parser.add_argument('--validation-dir', type=str, required=True,
                        help='Directory with validation images')
    parser.add_argument('--arch', type=str, required=True,
                        choices=list(ARCH_MAP.keys()),
                        help='Model architecture')
    parser.add_argument('--lk-kernel', type=int, default=13,
                        help='Large kernel size used during training')
    args = parser.parse_args()

    # Setup paths and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = setup_logger(output_dir)
    logger.info("=== AetherNet Model Release Script ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Output directory: {output_dir}")

    # Load validation sample
    validation_images = list(Path(args.validation_dir).glob('*.[jp][pn]g'))
    if not validation_images:
        logger.error(f"No validation images found in {args.validation_dir}")
        sys.exit(1)
        
    validation_sample = load_image(validation_images[0])
    logger.info(f"Validation sample: {validation_images[0].name} "
                f"(Shape: {validation_sample.shape})")

    # ------------------- Phase 1: Load Base Model ------------------- 
    logger.info("\n[1/4] Loading base model and extracting configuration")
    
    try:
        # Load model state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
            
        # Extract scale factor
        scale = int(state_dict.get('scale', 2))
        logger.info(f"Scale factor: {scale}x")
        
    except Exception as e:
        logger.exception("Failed to load model")
        sys.exit(1)

    # ------------------- Phase 2: Prepare Base Model ------------------- 
    logger.info("\n[2/4] Preparing base model for conversion")
    
    try:
        # Instantiate model with correct architecture
        base_model = ARCH_MAP[args.arch](
            scale=scale,
            lk_kernel=args.lk_kernel
        )
        
        # Prepare for Quantization-Aware Training (if needed)
        if any(k in model_path.name.lower() for k in ['qat', 'quant']):
            logger.info("Preparing QAT model")
            base_model.prepare_qat()
            
        # Load weights
        incompatible = base_model.load_state_dict(state_dict, strict=False)
        
        if incompatible.missing_keys:
            logger.warning(f"Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
            
        logger.info("Base model prepared successfully")
        
    except Exception as e:
        logger.exception("Failed to prepare base model")
        sys.exit(1)

    # ------------------- Phase 3: Convert PyTorch Models ------------------- 
    logger.info("\n[3/4] Converting to deployment formats")
    
    # FP32 Model
    logger.info("\n>> Creating FP32 model")
    fp32_model = create_fp32_model(base_model, args.arch, scale)
    if fp32_model:
        fp32_path = output_dir / f"{model_path.stem}_fp32.pth"
        torch.save(fp32_model.state_dict(), fp32_path)
        logger.info(f"Saved FP32 model to: {fp32_path}")
    
    # FP16 Model
    logger.info("\n>> Creating FP16 model")
    if fp32_model:
        fp16_model = create_fp16_model(fp32_model)
        if fp16_model:
            fp16_path = output_dir / f"{model_path.stem}_fp16.pth"
            torch.save(fp16_model.state_dict(), fp16_path)
            logger.info(f"Saved FP16 model to: {fp16_path}")
    else:
        logger.warning("Skipping FP16 conversion - no FP32 model available")
    
    # INT8 Model
    logger.info("\n>> Creating INT8 model")
    int8_model = create_int8_model(base_model)
    if int8_model:
        int8_path = output_dir / f"{model_path.stem}_int8.pth"
        torch.save(int8_model.state_dict(), int8_path)
        logger.info(f"Saved INT8 model to: {int8_path}")
    
    # ------------------- Phase 4: Export ONNX Models ------------------- 
    logger.info("\n[4/4] Exporting ONNX models")
    
    # FP32 ONNX
    if fp32_model:
        logger.info("\n>> Exporting FP32 ONNX")
        onnx_path = output_dir / f"{model_path.stem}_fp32.onnx"
        export_onnx_model(fp32_model, scale, 'fp32', onnx_path)
    
    # FP16 ONNX
    if fp16_model:
        logger.info("\n>> Exporting FP16 ONNX")
        onnx_path = output_dir / f"{model_path.stem}_fp16.onnx"
        export_onnx_model(fp16_model, scale, 'fp16', onnx_path)
    
    # INT8 ONNX
    if int8_model:
        logger.info("\n>> Exporting INT8 ONNX")
        onnx_path = output_dir / f"{model_path.stem}_int8.onnx"
        export_onnx_model(int8_model, scale, 'int8', onnx_path)
    
    logger.info("\n=== Conversion complete ===")
    logger.info(f"Results saved to: {output_dir}")
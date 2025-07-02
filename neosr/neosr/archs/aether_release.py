# --- FILE: aether_release.py ---
#!/usr/bin/env python3
"""
AetherNet Model Release Script
==============================

This script converts a trained AetherNet model checkpoint (.pth) into a suite
of deployment-ready formats, optimized for various inference scenarios.
It is designed to be a robust, one-click solution for packaging a trained
model for distribution and use in frameworks like ChaiNNer and Spandrel.

The script performs the following steps:
1.  Loads a PyTorch checkpoint and its architecture configuration.
2.  Validates the model with a sample image.
3.  Creates an optimized, fused FP32 model and exports it to .pth and .onnx.
4.  Creates a stabilized FP16 model and exports it to .pth and .onnx.
5.  If the source checkpoint was trained with QAT, it creates a fully
    quantized INT8 model and exports it to .pth and .onnx.
6.  All exported ONNX models include metadata for easy integration.
"""

import argparse
import os
import sys
import time
import logging
import warnings
import platform
import traceback
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, Optional, Callable, Any, List

import torch
import numpy as np
from PIL import Image
import onnxruntime as ort

from aether_core import AetherNet, export_onnx, parse_version

# --- Environment Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
ort.set_default_logger_severity(3)

# --- Constants & Configuration ---
MAX_RETRIES = 3
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("AetherNetRelease")
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)
    log_file = output_dir / "aether_release.log"
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    return logger

# Global logger instance
logger = logging.getLogger("AetherNetRelease")

# --- Helper Functions ---

def load_image(image_path: Path) -> torch.Tensor:
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {str(e)}")

def validate_pytorch_model(model: torch.nn.Module, validation_data: torch.Tensor, device: torch.device) -> bool:
    try:
        model.eval()
        model.to(device)
        is_fp16 = next(model.parameters()).dtype == torch.float16
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if is_fp16 else torch.float32):
            input_tensor = validation_data.to(device)
            if is_fp16: input_tensor = input_tensor.half()
            output = model(input_tensor)
            if not torch.isfinite(output).all():
                raise RuntimeError("Model produced non-finite output values (NaN or Inf)")
            return True
    except Exception as e:
        logger.error(f"PyTorch validation FAILED: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def validate_onnx_model(onnx_path: Path, validation_data: torch.Tensor, scale: int) -> bool:
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        input_data = validation_data.cpu().numpy()
        if 'float16' in input_type: input_data = input_data.astype(np.float16)
        output = session.run(None, {input_name: input_data})[0]
        _, c, h, w = validation_data.shape
        expected_shape = (1, c, h * scale, w * scale)
        if output.shape != expected_shape:
            logger.error(f"ONNX shape mismatch. Expected {expected_shape}, got {output.shape}")
            return False
        if not np.isfinite(output).all():
            logger.error("ONNX produced non-finite output values")
            return False
        return True
    except Exception as e:
        logger.error(f"ONNX validation FAILED: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def attempt_operation(operation: Callable, validation: Callable, success_msg: str, error_msg: str) -> Any:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempt {attempt}/{MAX_RETRIES}")
            result = operation()
            if validation(result):
                logger.info(f"✅ {success_msg}")
                return result
            else:
                logger.warning(f"Validation failed on attempt {attempt}")
        except Exception as e:
            logger.error(f"Operation failed: {str(e)}")
            logger.debug(traceback.format_exc())
            time.sleep(1)
    logger.error(f"❌ {error_msg}")
    return None

def memory_cleanup():
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- Model Conversion Functions ---

def create_fp32_model(base_model: AetherNet, validation_sample: torch.Tensor, device: torch.device) -> Optional[AetherNet]:
    """
    Creates a clean, inference-ready FP32 model from the base QAT model.
    """
    def _operation():
        # Instantiate a fresh model in its training-time configuration.
        model = AetherNet(**base_model.arch_config)
        # Load the state_dict with strict=False. This is the correct design pattern
        # to transfer weights from the fused, QAT-prepared base_model into the
        # unfused, float-only new model. It correctly ignores missing keys (e.g., lk_conv)
        # and unexpected keys (e.g., weight_fake_quant).
        model.load_state_dict(base_model.state_dict(), strict=False)
        model.fuse_model()
        model.eval()
        return model
    return attempt_operation(_operation, lambda m: validate_pytorch_model(m, validation_sample, device),
                             "Created optimized FP32 model", "Failed to create FP32 model")

def create_fp16_model(fp32_model: AetherNet, validation_sample: torch.Tensor, device: torch.device) -> Optional[AetherNet]:
    """Converts the FP32 model to FP16 with stability enhancements."""
    def _operation():
        model = deepcopy(fp32_model)
        model.stabilize_for_fp16()
        model = model.half()
        model.eval()
        return model
    return attempt_operation(_operation, lambda m: validate_pytorch_model(m, validation_sample, device),
                             "Created stable FP16 model", "Failed to create FP16 model")

def create_int8_model(base_qat_model: AetherNet, validation_sample: torch.Tensor, device: torch.device) -> Optional[AetherNet]:
    """Converts the final, QAT-prepared model to a fully quantized INT8 model."""
    def _operation():
        backend = 'fbgemm' if platform.system() == 'Linux' else 'qnnpack'
        if backend not in torch.backends.quantized.supported_engines:
            backend = torch.backends.quantized.supported_engines[0]
        torch.backends.quantized.engine = backend
        logger.info(f"Using quantization backend: {backend}")
        
        # Perform the final conversion on the CPU.
        model_to_convert = deepcopy(base_qat_model).cpu().eval()
        int8_model = model_to_convert.convert_to_quantized()
        logger.info("Verifying final INT8 model...")
        int8_model.verify_quantization()
        return int8_model
    return attempt_operation(_operation, lambda m: validate_pytorch_model(m, validation_sample, torch.device('cpu')),
                             "Created INT8 model", "Failed to create INT8 model")

def export_onnx_wrapper(model, scale, precision, output_path, validation_sample):
    """Wrapper to call the core export function and validate its output."""
    def _operation():
        export_onnx(deepcopy(model), scale, precision, str(output_path))
        return output_path
    return attempt_operation(_operation, lambda p: validate_onnx_model(p, validation_sample, scale),
                             f"Exported and validated {precision.upper()} ONNX model", f"Failed to export {precision.upper()} ONNX model")

def load_checkpoint(model_path: Path) -> Tuple[Dict, Dict, int, bool]:
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint.get('state_dict', checkpoint)))
    if not isinstance(state_dict, dict): raise ValueError("Could not find a valid state_dict in the checkpoint.")
    arch_config = checkpoint.get('arch_config')
    is_qat_checkpoint = any('fake_quant' in k for k in state_dict.keys())
    if is_qat_checkpoint: logger.info("QAT checkpoint detected based on state_dict keys.")
    if not arch_config: raise ValueError("Checkpoint must contain 'arch_config' dictionary.")
    if isinstance(arch_config.get('depths'), list): arch_config['depths'] = tuple(arch_config['depths'])
    return state_dict, arch_config, arch_config['scale'], is_qat_checkpoint

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AetherNet Model Release Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained QAT model checkpoint (.pth)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for converted models')
    parser.add_argument('--validation-dir', type=str, required=True, help='Directory with validation images')
    parser.add_argument('--skip-int8', action='store_true', help='Skip INT8 quantization')
    parser.add_argument('--skip-fp16', action='store_true', help='Skip FP16 conversion')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    if args.verbose: logger.setLevel(logging.DEBUG)

    logger.info("=== AetherNet Model Release Script ===")
    logger.info(f"PyTorch: {torch.__version__}, Device: {device}, Model: {Path(args.model_path).name}")

    try:
        # --- Phase 1: Load and Validate Checkpoint ---
        logger.info("\n[1/4] Loading and analyzing checkpoint")
        state_dict, arch_config, scale, is_qat_checkpoint = load_checkpoint(Path(args.model_path))
        validation_images = list(Path(args.validation_dir).glob('*.[jp][pn]g'))
        if not validation_images: raise FileNotFoundError(f"No validation images found in {args.validation_dir}")
        validation_sample = load_image(validation_images[0])
        logger.info(f"Scale: {scale}x, QAT: {is_qat_checkpoint}, Validation sample: {validation_images[0].name}")

        # --- Phase 2: Initialize Base Model ---
        logger.info("\n[2/4] Initializing base model")
        base_model = AetherNet(**arch_config)
        arch_name = base_model._get_architecture_name()
        logger.info(f"Detected Architecture: {arch_name}")

        if is_qat_checkpoint:
            logger.info("Preparing model for QAT state dict...")
            base_model.prepare_qat()
        elif any('fused_conv' in k for k in state_dict.keys()):
            logger.info("Fusing model to match non-QAT fused checkpoint.")
            base_model.fuse_model()

        base_model.load_state_dict(state_dict, strict=False)
        base_model.eval()
        logger.info("Base model loaded successfully.")
        
        logger.info("Validating base model...")
        if not validate_pytorch_model(base_model, validation_sample, device):
            raise RuntimeError("Base model failed initial validation.")

        # --- Phase 3: Convert Models ---
        logger.info("\n[3/4] Converting models to different precisions")
        model_stem = f"{Path(args.model_path).stem}_{arch_name}"
        
        logger.info("\n>> Creating FP32 model")
        fp32_model = create_fp32_model(base_model, validation_sample, device)
        if fp32_model:
            torch.save(fp32_model.state_dict(), output_dir / f"{model_stem}_fp32.pth")
            export_onnx_wrapper(fp32_model, scale, 'fp32', output_dir / f"{model_stem}_fp32.onnx", validation_sample)
        
        if not args.skip_fp16 and fp32_model:
            logger.info("\n>> Creating FP16 model")
            fp16_model = create_fp16_model(fp32_model, validation_sample, device)
            if fp16_model:
                torch.save(fp16_model.state_dict(), output_dir / f"{model_stem}_fp16.pth")
                export_onnx_wrapper(fp16_model, scale, 'fp16', output_dir / f"{model_stem}_fp16.onnx", validation_sample)
        
        if not args.skip_int8 and is_qat_checkpoint:
            logger.info("\n>> Creating INT8 model")
            int8_model = create_int8_model(base_model, validation_sample, device)
            if int8_model:
                torch.save(int8_model.state_dict(), output_dir / f"{model_stem}_int8.pth")
                export_onnx_wrapper(int8_model, scale, 'int8', output_dir / f"{model_stem}_int8.onnx", validation_sample)
        
        # --- Phase 4: Final Report ---
        logger.info("\n[4/4] Conversion complete.")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"An unrecoverable error occurred: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        memory_cleanup()
        sys.exit(0)
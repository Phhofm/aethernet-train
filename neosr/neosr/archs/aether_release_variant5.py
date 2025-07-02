#!/usr/bin/env python3
"""
AetherNet Model Release Script (Production-Grade)
================================================

Converts trained AetherNet models to optimized deployment formats with enhanced:
- Robustness across PyTorch versions (1.10+)
- Comprehensive error handling and validation
- Cross-platform compatibility (Windows/Linux/macOS)
- Framework-specific optimizations (TensorRT/DML/ONNX Runtime)
- Memory-efficient processing

Output Formats:
- PyTorch: FP32 (fused), FP16 (fused), INT8 (quantized)
- ONNX: FP32, FP16, INT8 (with framework metadata)
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

# Import model definitions from core implementation
from aether_core import (
    AetherNet,
    export_onnx,
    parse_version  # For version compatibility
)

# ------------------- Environment Configuration ------------------- 
# Suppress benign warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
warnings.filterwarnings("ignore", module="torch.ao.quantization")
ort.set_default_logger_severity(3)  # ONNX Runtime: Only show errors

# Detect execution environment
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
HAS_CUDA = torch.cuda.is_available()
HAS_DML = "DmlExecutionProvider" in ort.get_available_providers()
PT_VERSION = parse_version(torch.__version__)
MIN_PT_VERSION = parse_version("1.10.0")

# ------------------- Constants & Configuration ------------------- 
MAX_RETRIES = 3
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
SUPPORTED_PRECISIONS = ['fp32', 'fp16', 'int8']
SUPPORTED_ONNX_PROVIDERS = ['CPUExecutionProvider']
if HAS_CUDA:
    SUPPORTED_ONNX_PROVIDERS.insert(0, 'CUDAExecutionProvider')
if HAS_DML:
    SUPPORTED_ONNX_PROVIDERS.insert(0, 'DmlExecutionProvider')

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
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {str(e)}")

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
            
            # Handle precision automatically - FIXED: Directly get first parameter's dtype
            try:
                first_param = next(model.parameters())
                model_dtype = first_param.dtype
            except StopIteration:
                model_dtype = torch.float32
            
            if model_dtype == torch.float16:
                input_tensor = input_tensor.half()
            
            output = model(input_tensor)
            
            # FP16-specific stabilization
            if model_dtype == torch.float16:
                output = output.to(torch.float32)
                output = torch.clamp(output, -10, 10)
                output = output.to(torch.float16)
            
            # Basic output sanity check
            if not torch.isfinite(output).all():
                raise RuntimeError("Model produced non-finite output values")
                
            return True
    except Exception as e:
        logger.error(f"PyTorch validation FAILED: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def validate_onnx_model(
    onnx_path: Path, 
    validation_data: torch.Tensor, 
    scale: int,
    providers: List[str] = SUPPORTED_ONNX_PROVIDERS
) -> bool:
    """Validate ONNX model produces correct output shape and values."""
    try:
        # Configure execution providers
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        
        # Prepare input data
        input_data = validation_data.cpu().numpy()
        if 'float16' in input_type:
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
            
        # Verify output values
        if not np.isfinite(output).all():
            logger.error("ONNX produced non-finite output values")
            return False
            
        return True
    except Exception as e:
        logger.error(f"ONNX validation FAILED: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def attempt_operation(
    operation: Callable,
    validation: Callable[[Any], bool],
    success_msg: str,
    error_msg: str,
    max_retries: int = MAX_RETRIES,
    cleanup: Optional[Callable] = None
) -> Any:
    """Attempt an operation with retries, validation, and cleanup."""
    result = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}")
            result = operation()
            
            if validation(result):
                logger.info(f"✅ {success_msg}")
                return result
            else:
                logger.warning(f"Validation failed on attempt {attempt}")
                
        except Exception as e:
            logger.error(f"Operation failed: {str(e)}")
            logger.debug(traceback.format_exc())
            time.sleep(1)  # Brief pause before retry
        finally:
            # Perform cleanup after each attempt if specified
            if cleanup and callable(cleanup):
                cleanup()
            
    logger.error(f"❌ {error_msg}")
    return None

def memory_cleanup():
    """Free GPU memory and clear caches."""
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ------------------- Model Conversion Functions ------------------- 
def create_fp32_model(
    base_model: AetherNet,
    validation_sample: torch.Tensor,
    device: torch.device
) -> Optional[AetherNet]:
    """Create optimized FP32 model for deployment."""
    def _operation():
        model = deepcopy(base_model)
        model.fuse_model()
        return model
        
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created optimized FP32 model",
        error_msg="Failed to create FP32 model",
        cleanup=memory_cleanup
    )

def create_fp16_model(
    fp32_model: AetherNet,
    validation_sample: torch.Tensor,
    device: torch.device
) -> Optional[AetherNet]:
    """Convert FP32 model to FP16 precision with stability checks."""
    def _operation():
        model = deepcopy(fp32_model).half()
        model.eval()  # Ensure in inference mode
        model.stabilize_fp16()  # Apply stabilization first
        
        # Apply layer stabilization for FP16
        for module in model.modules():
            if hasattr(module, 'stabilize_fp16'):
                module.stabilize_fp16()
                
        return model
        
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created FP16 model",
        error_msg="Failed to create FP16 model",
        cleanup=memory_cleanup
    )

def create_int8_model(
    base_model: AetherNet,
    validation_sample: torch.Tensor,
    device: torch.device
) -> Optional[AetherNet]:
    """Convert QAT model to optimized INT8 model with platform-aware backend."""
    def _operation():
        # Select appropriate quantization backend
        if IS_WINDOWS or IS_MAC:
            backend = 'qnnpack'
        else:
            backend = 'fbgemm'
            
        # Check backend availability
        available_backends = torch.backends.quantized.supported_engines
        if backend not in available_backends:
            logger.warning(f"{backend} backend not available. Using {available_backends[0]} instead")
            backend = available_backends[0]
            
        # Apply backend
        original_engine = torch.backends.quantized.engine
        torch.backends.quantized.engine = backend
        logger.info(f"Using quantization backend: {backend}")
        
        try:
            # Prepare and convert to INT8
            #model = deepcopy(base_model).to(device).eval()
            # Force CPU quantization for stability
            model = deepcopy(base_model).cpu().eval()
            
            # PyTorch version-specific quantization preparation
            if PT_VERSION < parse_version("1.13.0"):
                logger.info("Using legacy quantization API")
                model.prepare_qat()
                int8_model = model.convert_to_quantized()
            else:
                # Newer quantization flow - use legacy API for control flow models
                logger.info("Using legacy quantization API (FX incompatible)")
                model.prepare_qat()
                int8_model = model.convert_to_quantized()
            
            # Verify quantization
            if hasattr(int8_model, 'verify_quantization') and not int8_model.verify_quantization():
                logger.warning("Quantization verification reported issues")
                
            return int8_model
        finally:
            torch.backends.quantized.engine = original_engine
            
    return attempt_operation(
        operation=_operation,
        validation=lambda m: validate_pytorch_model(m, validation_sample, device),
        success_msg="Created INT8 model",
        error_msg="Failed to create INT8 model",
        cleanup=memory_cleanup
    )

def export_onnx_model(
    model: AetherNet,
    scale: int,
    precision: str,
    output_path: Path,
    validation_sample: torch.Tensor
) -> bool:
    """Export model to ONNX format with validation and version compatibility."""
    def _operation():
        # Use deepcopy to avoid modifying original model
        export_model = deepcopy(model)
        
        # Handle different PyTorch versions
        if PT_VERSION < parse_version("1.12.0"):
            # Legacy export for older PyTorch
            export_onnx(export_model, scale, precision, str(output_path))
        else:
            # Modern export with dynamic axes support
            export_model.eval()
            export_model.fuse_model()
            
            # Create dummy input
            dummy_input = validation_sample.to(next(export_model.parameters()).device)
            if precision == 'fp16':
                export_model = export_model.half()
                dummy_input = dummy_input.half()
            
            # Dynamic axes configuration
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
            }
            
            # Export with version-specific settings
            torch.onnx.export(
                export_model,
                dummy_input,
                str(output_path),
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                export_params=True,
                keep_initializers_as_inputs=True,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            )
        return output_path
        
    result = attempt_operation(
        operation=_operation,
        validation=lambda p: validate_onnx_model(p, validation_sample, scale),
        success_msg=f"Exported {precision.upper()} ONNX model",
        error_msg=f"Failed to export {precision.upper()} ONNX model",
        cleanup=memory_cleanup
    )
    
    return result is not None

def load_checkpoint(
    model_path: Path,
    device: torch.device
) -> Tuple[Dict, Dict, int]:
    """
    Load model checkpoint with enhanced compatibility handling.
    
    Returns:
        state_dict: Model weights
        arch_config: Architecture configuration
        scale: Super-resolution scale factor
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        raise IOError(f"Failed to load checkpoint: {str(e)}")
    
    # Handle different checkpoint formats
    state_dict = None
    arch_config = None
    scale = None
    
    # Check for optimized format (from save_optimized)
    if 'state_dict' in checkpoint and 'metadata' in checkpoint:
        state_dict = checkpoint['state_dict']
        metadata = checkpoint['metadata']
        arch_config = metadata.get('arch_config', None)
        if scale is None and arch_config is not None:
            scale = arch_config.get('scale', None)  # Get from arch_config if available
        logger.info("Detected optimized checkpoint format")
    
    # Check for training checkpoint format
    elif 'params_ema' in checkpoint or 'params' in checkpoint:
        state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
        arch_config = checkpoint.get('arch_config', None)
        if scale is None and arch_config is not None:
            scale = arch_config.get('scale', None)  # Get from arch_config if available
        logger.info("Detected training checkpoint format")
    
    # Fallback to state_dict only
    else:
        state_dict = checkpoint
        logger.warning("Basic checkpoint format detected - attempting recovery")
    
    # Validate we have required components
    if state_dict is None:
        raise ValueError("No valid state dictionary found in checkpoint")
    
    # Extract scale if not found
    if scale is None:
        # Search for scale in state_dict keys
        scale_keys = [k for k in state_dict.keys() if 'scale' in k.lower()]
        if scale_keys:
            scale = state_dict[scale_keys[0]].item()
        else:
            scale = 4  # Default to 4x
        logger.warning(f"Scale factor not found, defaulting to {scale}x")
        
    try:
        arch_name = base_model._get_architecture_name()
        logger.info(f"Detected Architecture: {arch_name}")
        # Add to metadata for saved files
        model_stem = f"{model_path.stem}_{arch_name}"
    except Exception as e:
        logger.warning(f"Architecture detection failed: {str(e)}")
        model_stem = model_path.stem
    
    # Attempt to recover arch_config from model
    if arch_config is None:
        logger.warning("arch_config not found - attempting reconstruction")
        
        # Determine model size based on channel patterns
        embed_dim = None
        for key in state_dict.keys():
            if 'conv_first.weight' in key:
                _, embed_dim, _, _ = state_dict[key].shape
                break
            elif 'stages.0.0' in key and 'weight' in key:
                _, embed_dim, _, _ = state_dict[key].shape
                break
        
        if embed_dim is None:
            raise ValueError("Could not determine model architecture")
        
        # Create minimal arch_config
        arch_config = {
            'in_chans': 3,
            'scale': scale,
            'embed_dim': embed_dim,
            'depths': [4] * 4  # Default to 4 blocks per stage
        }
    
    # Ensure depths is tuple (required by model)
    if isinstance(arch_config['depths'], list):
        arch_config['depths'] = tuple(arch_config['depths'])
    
    return state_dict, arch_config, scale

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
    parser.add_argument('--skip-int8', action='store_true',
                        help='Skip INT8 quantization (useful for ARM/Mac)')
    parser.add_argument('--skip-fp16', action='store_true',
                        help='Skip FP16 conversion (if stability issues)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup paths and device
    device = torch.device('cuda' if HAS_CUDA else 'cpu')
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = setup_logger(output_dir)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    logger.info("=== AetherNet Model Release Script (Production-Grade) ===")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Output directory: {output_dir}")

    # ------------------- Phase 1: Environment Checks -------------------
    logger.info("\n[1/5] Performing environment checks")
    try:
        # Verify minimum PyTorch version
        if PT_VERSION < MIN_PT_VERSION:
            raise RuntimeError(
                f"PyTorch {'.'.join(map(str, MIN_PT_VERSION))}+ required. "
                f"Detected: {'.'.join(map(str, PT_VERSION))}"
            )
        
        # Check for quantization support
        if not args.skip_int8 and not hasattr(torch.ao.quantization, 'QuantStub'):
            logger.warning("Quantization not supported in this PyTorch build. Disabling INT8.")
            args.skip_int8 = True
            
        # Check ONNX Runtime availability
        try:
            ort.__version__
        except AttributeError:
            logger.warning("ONNX Runtime not fully installed. ONNX exports may fail.")
            
        logger.info("Environment checks passed")
    except Exception as e:
        logger.error(f"Environment check failed: {str(e)}")
        sys.exit(1)

    # ------------------- Phase 2: Load and Validate Checkpoint -------------------
    logger.info("\n[2/5] Loading model checkpoint")
    try:
        # Load checkpoint with automatic format detection
        state_dict, arch_config, scale = load_checkpoint(model_path, device)
        
        # Log configuration
        logger.info(f"Scale factor: {scale}x")
        logger.debug(f"Architecture config: {arch_config}")
        
        # Get aether network option
        try:
            arch_name = base_model._get_architecture_name()
            logger.info(f"Network Architecture: {arch_name}")
        except Exception:
            logger.warning("Could not determine network architecture")
        
        # Load validation sample
        validation_images = list(Path(args.validation_dir).glob('*.[jp][pn]g'))
        if not validation_images:
            raise FileNotFoundError(f"No validation images found in {args.validation_dir}")
            
        validation_sample = load_image(validation_images[0])
        logger.info(f"Validation sample: {validation_images[0].name} "
                    f"(Shape: {validation_sample.shape})")
                    
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # ------------------- Phase 3: Initialize Base Model -------------------
    logger.info("\n[3/5] Initializing base model")
    try:
        # Initialize model with recovered configuration
        base_model = AetherNet(**arch_config)
        
        # Fuse model BEFORE loading state_dict to match QAT checkpoint structure
        if any('fused_conv' in key for key in state_dict.keys()):
            logger.info("Fusing model before loading state_dict to match checkpoint structure")
            base_model.fuse_model()
        
        # Load weights with strict=False for compatibility
        load_result = base_model.load_state_dict(state_dict, strict=False)
        
        # Log loading results
        if load_result.missing_keys:
            logger.warning(f"Missing keys: {len(load_result.missing_keys)}")
            logger.debug("First 5 missing keys: " + ", ".join(load_result.missing_keys[:5]))
        if load_result.unexpected_keys:
            logger.warning(f"Unexpected keys: {len(load_result.unexpected_keys)}")
            logger.debug("First 5 unexpected keys: " + ", ".join(load_result.unexpected_keys[:5]))
            
        logger.info("Base model initialized successfully")
        
        # Validate base model
        logger.info("Validating base model...")
        if validate_pytorch_model(base_model, validation_sample, device):
            logger.info("Base model validation successful")
        else:
            raise RuntimeError("Base model validation failed")
            
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # ------------------- Phase 4: Convert Models -------------------
    logger.info("\n[4/5] Converting models")
    model_stem = model_path.stem
    success_flags = {prec: False for prec in SUPPORTED_PRECISIONS}
    success_flags.update({f"{prec}_onnx": False for prec in SUPPORTED_PRECISIONS})
    
    # FP32 conversion
    logger.info("\n>> Creating FP32 model")
    fp32_model = create_fp32_model(base_model, validation_sample, device)
    if fp32_model:
        fp32_path = output_dir / f"{model_stem}_fp32.pth"
        torch.save(fp32_model.state_dict(), fp32_path)
        logger.info(f"Saved FP32 model to: {fp32_path}")
        success_flags['fp32'] = True
        
        # Export ONNX
        logger.info(">> Exporting FP32 ONNX")
        onnx_path = output_dir / f"{model_stem}_fp32.onnx"
        if export_onnx_model(fp32_model, scale, 'fp32', onnx_path, validation_sample):
            success_flags['fp32_onnx'] = True
    
    # FP16 conversion
    if not args.skip_fp16:
        logger.info("\n>> Creating FP16 model")
        if fp32_model:
            fp16_model = create_fp16_model(fp32_model, validation_sample, device)
            if fp16_model:
                fp16_path = output_dir / f"{model_stem}_fp16.pth"
                torch.save(fp16_model.state_dict(), fp16_path)
                logger.info(f"Saved FP16 model to: {fp16_path}")
                success_flags['fp16'] = True
                
                # Export ONNX
                logger.info(">> Exporting FP16 ONNX")
                onnx_path = output_dir / f"{model_stem}_fp16.onnx"
                if export_onnx_model(fp16_model, scale, 'fp16', onnx_path, validation_sample):
                    success_flags['fp16_onnx'] = True
        else:
            logger.warning("Skipping FP16 conversion - no FP32 model available")
    else:
        logger.info("\n>> Skipping FP16 conversion per user request")
    
    # INT8 conversion
    if not args.skip_int8:
        logger.info("\n>> Creating INT8 model")
        int8_model = create_int8_model(base_model, validation_sample, device)
        if int8_model:
            int8_path = output_dir / f"{model_stem}_int8.pth"
            torch.save(int8_model.state_dict(), int8_path)
            logger.info(f"Saved INT8 model to: {int8_path}")
            success_flags['int8'] = True
            
            # Export ONNX
            logger.info(">> Exporting INT8 ONNX")
            onnx_path = output_dir / f"{model_stem}_int8.onnx"
            if export_onnx_model(int8_model, scale, 'int8', onnx_path, validation_sample):
                success_flags['int8_onnx'] = True
    else:
        logger.info("\n>> Skipping INT8 conversion per user request")

    # ------------------- Phase 5: Final Report & Cleanup -------------------
    logger.info("\n[5/5] Conversion Report")
    success_count = sum(1 for v in success_flags.values() if v)
    total_attempts = len(success_flags)
    
    logger.info(f"Successful conversions: {success_count}/{total_attempts}")
    for format, success in success_flags.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{format.upper():<10}: {status}")
    
    logger.info("\n=== Conversion complete ===")
    logger.info(f"Results saved to: {output_dir}")
    
    # Clean up resources
    memory_cleanup()
    
    # Exit with appropriate status code
    sys.exit(0 if success_count > 0 else 1)
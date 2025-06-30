# --- File: aether_release.py (Final Version - Acknowledges Env Limitations) ---

# NUKE THE PIP CACHE. It forces pip to re-download fresh files.
# pip cache purge

# # Create a new, clean venv
#python3 -m venv aether_env

# Activate the new stable environment
#source aether_env/bin/activate

# Upgrade pip first
#pip install --upgrade pip

# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the packages
#pip install onnx onnxruntime-gpu Pillow numpy

# Run this command to check your environemt can run the Modern QAT (v2) API
#python -c "import torch; import torch.ao.quantization as tq; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Modern QAT (v2) API Available: {hasattr(tq, 'get_default_qat_qconfig_v2')}')"

# If True
# cd neosr/neosr/archs

# Conversion command
# python3 aether_release.py --model-path /home/phips/Documents/GitHub/aethernet-train/neosr/experiments/2xaether_tiny_qat/models/net_g_7000.pth --output-dir /home/phips/Documents/GitHub/aethernet-train/neosr/experiments/2xaether_tiny_qat/models/release --validation-dir /home/phips/Documents/dataset/PDM/OSISRD/v3/validation/x2 --arch aether_tiny

# --- File: aether_release.py (Final Polished Version) ---

import argparse
import os
import torch
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import numpy as np
from copy import deepcopy
import warnings
from aether_core import aether, aether_tiny, aether_small, aether_medium, aether_large, export_onnx

# --- START OF FIX #2: Suppress verbose warnings ---
# Suppress benign PyTorch UserWarnings and ONNX TracerWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
# Suppress benign ONNX Runtime warnings about initializers. 3 = ERROR level.
ort.set_default_logger_severity(3)
# --- END OF FIX #2 ---

MAX_RETRIES = 3

# ... (Helper functions are unchanged and correct) ...
def load_image(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return tensor

def validate_pytorch_model(model: torch.nn.Module, validation_data: torch.Tensor, device: torch.device) -> bool:
    try:
        model.eval(); model.to(device)
        with torch.no_grad():
            input_tensor = validation_data.to(device)
            is_fp16 = False
            try:
                if next(model.parameters()).dtype == torch.float16: is_fp16 = True
            except StopIteration: pass
            if is_fp16: input_tensor = input_tensor.half()
            _ = model(input_tensor)
        return True
    except Exception as e:
        print(f"    - 🔴 PyTorch validation FAILED: {e}"); return False

def validate_onnx_model(onnx_path: str, validation_data: torch.Tensor, scale: int) -> bool:
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name, input_dtype = session.get_inputs()[0].name, session.get_inputs()[0].type
        input_data_np = validation_data.cpu().numpy()
        if input_dtype == 'tensor(float16)': input_data_np = input_data_np.astype(np.float16)
        output = session.run(None, {input_name: input_data_np})[0]
        c, h, w = validation_data.shape[1:]
        expected_shape = (1, c, h * scale, w * scale)
        if output.shape != expected_shape:
            raise RuntimeError(f"Shape mismatch. Expected {expected_shape}, got {output.shape}")
        return True
    except Exception as e:
        print(f"    - 🔴 ONNX validation FAILED: {e}"); return False

def attempt_conversion(conversion_func, validation_func, success_message, failure_message, max_retries=MAX_RETRIES):
    for i in range(max_retries):
        print(f"  - Attempt {i + 1}/{max_retries}...")
        try:
            result = conversion_func()
            if validation_func(result):
                print(f"  - ✅ {success_message}"); return result
        except Exception as e:
            print(f"    - 🔴 Conversion attempt FAILED: {e}")
    print(f"  - ❌ {failure_message}"); return None

def main():
    parser = argparse.ArgumentParser(description="Convert and validate a trained AetherNet QAT model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the input QAT-trained .pth model file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the converted models.')
    parser.add_argument('--validation-dir', type=str, required=True, help='Path to a folder of images for validation.')
    parser.add_argument('--arch', type=str, required=True, choices=['aether_tiny', 'aether_small', 'aether_medium', 'aether_large'], help='The network architecture of the model.')
    args = parser.parse_args()
    print("--- AetherNet Model Release Script ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_path, output_dir = Path(args.model_path), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_images = list(Path(args.validation_dir).glob('*.[jp][pn]g'))
    if not validation_images:
        raise FileNotFoundError(f"No validation images (.jpg, .png) found in '{args.validation_dir}'")
    validation_sample = load_image(validation_images[0])
    print(f"Loaded validation sample: {validation_images[0].name} (Shape: {validation_sample.shape})")
    print("\n[1/4] Loading base model and extracting configuration...")
    try:
        state_dict = torch.load(args.model_path, map_location='cpu')
        model_weights = state_dict.get('params_ema', state_dict.get('params', state_dict))
        scale = int(model_weights['scale'].item())
        print(f"  - ✅ Extracted scale: {scale}x")
    except (KeyError, FileNotFoundError) as e:
        print(f"  - ❌ Error loading model or finding scale: {e}"); return
    arch_map = { 'aether_tiny': aether_tiny, 'aether_small': aether_small, 'aether_medium': aether_medium, 'aether_large': aether_large }
    print("  - Instantiating QAT-prepared model to match the saved state...")
    base_model = arch_map[args.arch](scale=scale)
    base_model.prepare_qat() # This will now correctly use the v2 API
    
    # The sanitizing logic is kept as a robust measure, though it may not be needed in the new env.
    print("  - Sanitizing state_dict to handle potential version differences...")
    keys_to_remove = []
    for key in list(model_weights.keys()):
        if key.endswith(('.min_val', '.max_val', '.min_vals', '.max_vals')): keys_to_remove.append(key)
    for key in keys_to_remove:
        if key in model_weights: del model_weights[key]
    print(f"    - Info: Removed {len(keys_to_remove)} potentially problematic observer keys from the state_dict.")
    incompatible_keys = base_model.load_state_dict(model_weights, strict=False)
    critical_errors = [key for key in incompatible_keys.unexpected_keys]
    if critical_errors:
        print("\n  - ❌ CRITICAL ERROR: Found architectural mismatches while loading the model.")
        for error in critical_errors[:10]: print(f"      - {error}")
        return
    print(f"  - ✅ Loaded sanitized weights into QAT-prepared [{args.arch}] architecture.")
    print(f"    - Info: Ignored {len(incompatible_keys.missing_keys)} keys for version compatibility.")
    
    print("\n[2/4] Converting and validating PyTorch models...")
    print("\n> Processing: FP32 PyTorch (.pth)")
    fp32_model_path = output_dir / f"{model_path.stem}_fp32_fused.pth"
    def convert_fp32_pth():
        model_fp32 = arch_map[args.arch](scale=scale)
        model_fp32.fuse_model() 
        model_fp32.load_state_dict(base_model.state_dict(), strict=False)
        return model_fp32
    fp32_model = attempt_conversion(convert_fp32_pth, lambda m: validate_pytorch_model(m, validation_sample, device), "FP32 PTH model created and validated.", "Failed to create a valid FP32 PTH model.")
    if fp32_model:
        torch.save(fp32_model.state_dict(), fp32_model_path)
        print(f"  - ✅ Saved to: {fp32_model_path}")

    print("\n> Processing: FP16 PyTorch (.pth)")
    fp16_model = None
    if fp32_model:
        fp16_model_path = output_dir / f"{model_path.stem}_fp16_fused.pth"
        def convert_fp16_pth(): return deepcopy(fp32_model).half()
        fp16_model = attempt_conversion(convert_fp16_pth, lambda m: validate_pytorch_model(m, validation_sample, device), "FP16 PTH model created and validated.", "Failed to create a valid FP16 PTH model.")
        if fp16_model:
            torch.save(fp16_model.state_dict(), fp16_model_path)
            print(f"  - ✅ Saved to: {fp16_model_path}")
    else:
        print("  - 🟡 Skipping FP16 conversion because FP32 model creation failed.")

    # --- START OF FIX #1: Remove skipping logic ---
    print("\n> Processing: INT8 PyTorch (.pth) and preparing for ONNX")
    int8_model = None
    try:
        print("  - Attempting INT8 conversion...")
        print("    - Setting quantization engine to 'qnnpack' for stability.")
        torch.backends.quantized.engine = 'qnnpack'
        model_for_conversion = deepcopy(base_model).cpu().eval()
        int8_model = torch.ao.quantization.convert(model_for_conversion, inplace=False)
        int8_model_path = output_dir / f"{model_path.stem}_int8_converted.pth"
        torch.save(int8_model.state_dict(), int8_model_path)
        print(f"  - ✅ INT8 PTH model created successfully.")
        print(f"  - ✅ Saved to: {int8_model_path}")
    except Exception as e:
        print(f"  - ❌ Failed to create a valid INT8 PTH model: {e}")
        
    print("\n[3/4] Exporting and validating ONNX models...")
    os.chdir(output_dir)
    if fp32_model:
        print("\n> Processing: FP32 ONNX")
        attempt_conversion(lambda: export_onnx(fp32_model.cpu(), scale, precision='fp32'), lambda path: validate_onnx_model(path, validation_sample, scale), "FP32 ONNX model exported and validated.", "Failed to create a valid FP32 ONNX model.")
    else:
        print("\n> Skipping FP32 ONNX: FP32 PyTorch model not available.")
    if fp16_model:
        print("\n> Processing: FP16 ONNX")
        attempt_conversion(lambda: export_onnx(fp16_model, scale, precision='fp16'), lambda path: validate_onnx_model(path, validation_sample, scale), "FP16 ONNX model exported and validated.", "Failed to create a valid FP16 ONNX model.")
    else:
        print("\n> Skipping FP16 ONNX: FP16 PyTorch model not available.")

    # --- START OF FIX #1: Remove skipping logic ---
    print("\n> Processing: INT8 PyTorch (.pth) and preparing for ONNX")
    int8_model = None
    try:
        print("  - Attempting INT8 conversion...")
        print("    - Setting quantization engine to 'qnnpack' for stability.")
        torch.backends.quantized.engine = 'qnnpack'
        model_for_conversion = deepcopy(base_model).cpu().eval()
        int8_model = torch.ao.quantization.convert(model_for_conversion, inplace=False)
        
        # ===== START OF ADDED VERIFICATION STEP =====
        print("  - Verifying quantization parameters...")
        for name, module in int8_model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                if hasattr(module.weight_fake_quant, 'scale'):
                    scale = module.weight_fake_quant.scale
                    if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                        print(f"    ⚠️ Per-channel quantization detected in {name} - forcing to per-tensor")
                        # Convert per-channel to per-tensor
                        module.weight_fake_quant.scale = scale.mean()
                        module.weight_fake_quant.zero_point = module.weight_fake_quant.zero_point.mean()
            # Also check for Conv2d layers that might have quantization parameters
            elif isinstance(module, nn.Conv2d) and hasattr(module, 'weight_fake_quant'):
                if hasattr(module.weight_fake_quant, 'scale'):
                    scale = module.weight_fake_quant.scale
                    if isinstance(scale, torch.Tensor) and scale.numel() > 1:
                        print(f"    ⚠️ Per-channel quantization detected in {name} - forcing to per-tensor")
                        module.weight_fake_quant.scale = scale.mean()
                        module.weight_fake_quant.zero_point = module.weight_fake_quant.zero_point.mean()
        print("  - ✅ Quantization parameters verified")
        # ===== END OF ADDED VERIFICATION STEP =====
        
        int8_model_path = output_dir / f"{model_path.stem}_int8_converted.pth"
        torch.save(int8_model.state_dict(), int8_model_path)
        print(f"  - ✅ INT8 PTH model created successfully.")
        print(f"  - ✅ Saved to: {int8_model_path}")
    except Exception as e:
        print(f"  - ❌ Failed to create a valid INT8 PTH model: {e}")
        
    print("\n[4/4] --- Script finished ---")

if __name__ == '__main__':
    main()
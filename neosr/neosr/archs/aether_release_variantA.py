# --- File: aether_release.py (The Definitive, Correct Version) ---

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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
ort.set_default_logger_severity(3)
MAX_RETRIES = 3

# --- Helper functions are correct and unchanged ---
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

# --- main function with the final, correct logic ---
def main():
    # ... (arg parsing and setup is correct) ...
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
    base_model.prepare_qat() # This uses the corrected per-tensor version
    
    # We don't need to sanitize anymore, as prepare_qat now creates a compatible state
    base_model.load_state_dict(model_weights, strict=False)
    print(f"  - ✅ Loaded weights into QAT-prepared [{args.arch}] architecture.")

    print("\n[2/4] Converting and validating PyTorch models...")
    # ... FP32/FP16 logic is correct and unchanged ...
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

    # --- Create the REAL INT8 PTH file ---
    print("\n> Processing: INT8 PyTorch (.pth)")
    int8_model_for_pth = None
    try:
        print("  - Attempting INT8 conversion for PyTorch...")
        # We need the qnnpack engine for the convert() call to succeed
        torch.backends.quantized.engine = 'qnnpack'
        model_for_conversion = deepcopy(base_model).cpu().eval()
        int8_model_for_pth = torch.ao.quantization.convert(model_for_conversion, inplace=False)
        int8_model_path = output_dir / f"{model_path.stem}_int8_converted.pth"
        torch.save(int8_model_for_pth.state_dict(), int8_model_path)
        print(f"  - ✅ INT8 PTH model created successfully.")
        print(f"  - ✅ Saved to: {int8_model_path}")
    except Exception as e:
        print(f"  - ❌ Failed to create a valid INT8 PTH model: {e}")

    print("\n[3/4] Exporting and validating ONNX models...")
    os.chdir(output_dir)
    # ... FP32/FP16 ONNX logic is correct and unchanged ...
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
        
    # --- Create the REAL INT8 ONNX file ---
    print("\n> Processing: INT8 ONNX")
    try:
        # Convert to true quantized model
        quantized_model = base_model.convert_to_quantized()
        
        # Verify quantization
        print("  - Verifying quantization state...")
        quantized_model.verify_quantization()
        
        # Export
        attempt_conversion(
            lambda: export_onnx(quantized_model.cpu(), scale, precision='int8'),
            lambda path: validate_onnx_model(path, validation_sample, scale),
            "INT8 ONNX model exported and validated.",
            "Failed to create a valid INT8 ONNX model."
        )
    except Exception as e:
        print(f"  - ❌ INT8 ONNX conversion failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[4/4] --- Script finished ---")

if __name__ == '__main__':
    main()
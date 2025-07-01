import argparse
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
from copy import deepcopy

# Import model definition and export function from your core file
from aether_core import aether, aether_tiny, aether_small, aether_medium, aether_large, export_onnx

def load_image(image_path: Path) -> torch.Tensor:
    """Loads an image and converts it to a tensor."""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

def validate_pytorch_model(model, validation_data, device):
    """Runs a single inference to check for runtime errors."""
    try:
        model.to(device).eval()
        input_tensor = validation_data.to(device)
        if next(model.parameters()).dtype == torch.float16:
            input_tensor = input_tensor.half()
        with torch.no_grad():
            _ = model(input_tensor)
        return True
    except Exception as e:
        print(f"    - 🔴 PyTorch validation FAILED: {e}")
        return False

def validate_onnx_model(onnx_path, validation_data, scale):
    """Runs ONNX inference and checks output shape."""
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        input_dtype = session.get_inputs()[0].type
        input_data = validation_data.cpu().numpy()
        if input_dtype == 'tensor(float16)':
            input_data = input_data.astype(np.float16)
        output = session.run(None, {input_name: input_data})[0]
        _, c, h, w = validation_data.shape
        if output.shape == (1, c, h * scale, w * scale): return True
        print(f"    - 🔴 ONNX shape mismatch. Expected {(1, c, h * scale, w * scale)}, got {output.shape}")
        return False
    except Exception as e:
        print(f"    - 🔴 ONNX validation FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AetherNet Release Script: Converts a trained model to various deployment formats.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to the trained .pth model file.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save the converted models.")
    parser.add_argument("--validation-dir", required=True, type=str, help="Directory with validation images for testing the converted models.")
    parser.add_argument("--arch", required=True, type=str, choices=['aether_tiny', 'aether_small', 'aether_medium', 'aether_large'], help="Model architecture.")
    parser.add_argument("--lk_kernel", type=int, default=13, help="Large kernel size used during training. Defaults to 13.")
    args = parser.parse_args()
    
    print("--- AetherNet Model Release Script ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir, model_path = Path(args.output_dir), Path(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_sample = load_image(list(Path(args.validation_dir).glob('*.[jp][pn]g'))[0])
    print(f"Loaded validation sample (Shape: {validation_sample.shape})")

    print("\n[1/4] Loading base model and extracting configuration...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    model_weights = state_dict.get('params_ema', state_dict.get('params', state_dict))
    scale = int(model_weights.get('scale', 2))
    print(f"  - ✅ Extracted scale: {scale}x")

    arch_map = {'aether_tiny': aether_tiny, 'aether_small': aether_small, 'aether_medium': aether_medium, 'aether_large': aether_large}
    
    print("  - Instantiating base model...")
    # Instantiate the clean architecture, overriding defaults if needed (e.g., lk_kernel)
    base_model = arch_map[args.arch](scale=scale, lk_kernel=args.lk_kernel)
    base_model.load_state_dict(model_weights, strict=True)
    print(f"  - ✅ Loaded weights into [{args.arch}] architecture.")

    print("\n[2/4] Creating and validating PyTorch models...")
    
    # --- FP32 PTH ---
    fp32_model = deepcopy(base_model); fp32_model.fuse_model()
    fp32_path = output_dir / f"{model_path.stem}_fp32_fused.pth"
    torch.save(fp32_model.state_dict(), fp32_path)
    print(f"> Processing: FP32 PyTorch (.pth)\n  - ✅ Saved to: {fp32_path}")

    # --- FP16 PTH ---
    fp16_model = deepcopy(fp32_model).half()
    fp16_path = output_dir / f"{model_path.stem}_fp16_fused.pth"
    torch.save(fp16_model.state_dict(), fp16_path)
    print(f"> Processing: FP16 PyTorch (.pth)\n  - ✅ Saved to: {fp16_path}")

    # --- INT8 PTH ---
    print("> Processing: INT8 PyTorch (.pth)")
    qat_model = arch_map[args.arch](scale=scale, lk_kernel=args.lk_kernel)
    qat_model.load_state_dict(model_weights, strict=True)
    int8_model = qat_model.prepare_qat().convert_to_quantized()
    int8_path = output_dir / f"{model_path.stem}_int8_converted.pth"
    torch.save(int8_model.state_dict(), int8_path)
    print(f"  - ✅ Saved to: {int8_path}")

    print("\n[3/4] Exporting and validating ONNX models...")
    
    # --- FP32/FP16 ONNX ---
    for precision, model_instance in [("fp32", fp32_model), ("fp16", fp16_model)]:
        print(f"\n> Processing: {precision.upper()} ONNX")
        onnx_path = output_dir / f"{model_path.stem}_{precision}.onnx"
        export_onnx(deepcopy(model_instance), scale, precision, onnx_path)
        if not validate_onnx_model(onnx_path, validation_sample, scale):
            print(f"  - ❌ {precision.upper()} ONNX validation failed.")

    # --- INT8 ONNX ---
    print("\n> Processing: INT8 ONNX")
    int8_onnx_path = output_dir / f"{model_path.stem}_int8.onnx"
    export_onnx(int8_model, scale, "int8", int8_onnx_path)
    if not validate_onnx_model(int8_onnx_path, validation_sample, scale):
        print("  - ❌ INT8 ONNX validation failed.")

    print("\n[4/4] All processes finished.")

if __name__ == "__main__":
    main()
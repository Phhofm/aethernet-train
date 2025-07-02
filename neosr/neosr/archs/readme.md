# AetherNet: An Ultra-Fast Super-Resolution Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-%23EE4C2C?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.12+-%235A94DB?logo=onnx)](https://onnx.ai/)

AetherNet is a production-ready, high-performance Single-Image Super-Resolution (SISR) network designed for extreme speed and efficiency without sacrificing quality. It is built from the ground up to leverage modern deep learning techniques like structural reparameterization and Quantization-Aware Training (QAT) to deliver state-of-the-art inference speeds on a wide range of hardware.

This repository contains the core model architecture and a robust deployment script for converting trained models into highly optimized formats.

-   **Main Repository:** [https://github.com/Phhofm/aethernet](https://github.com/Phhofm/aethernet)
-   **Training Framework:** AetherNet can be trained using the NeoSR framework available at [https://github.com/Phhofm/aethernet-train](https://github.com/Phhofm/aethernet-train).

## The Name: AetherNet

The name "AetherNet" is inspired by the classical element 'aether' (or 'ether'), which was once thought to be a weightless, transparent substance that filled all of space. This name was chosen to reflect the network's core design goals: to be incredibly **lightweight** and **fast**, processing images so efficiently that it feels almost instantaneous, as if passing through the aether itself.

## Core Philosophy & Design

AetherNet was built to bridge the gap between academic research models, which often prioritize peak quality at any computational cost, and real-world applications, which demand speed and efficiency.

### 1. Structural Reparameterization for Zero-Cost Quality
At training time, AetherNet uses complex blocks with multiple parallel branches (e.g., large-kernel and small-kernel convolutions). These branches help the model learn more robust features and increase its representative power.

However, for inference, these parallel branches are mathematically **fused into a single, standard convolution layer**. This means you get the performance benefits of a much more complex architecture with the inference speed of a simple, shallow network. The quality boost is effectively "free" at runtime.

### 2. Quantization-Aware Training (QAT) for INT8 Speed
Floating-point (FP32/FP16) arithmetic is the standard for training, but integer (INT8) arithmetic is significantly faster on modern CPUs and GPUs. Naively converting a model to INT8 (Post-Training Quantization) often leads to a severe drop in quality.

AetherNet is designed to be fully compatible with **Quantization-Aware Training**. The model learns to adapt to the constraints of 8-bit integers *during* the training process. This results in an INT8 model that is nearly identical in quality to its full-precision counterpart but with a major speed advantage.

### 3. Production-Ready Deployment
A great model is only useful if it can be deployed. AetherNet comes with a robust release script (`aether_release.py`) that automates the entire conversion process, creating a full suite of deployable artifacts from a single trained checkpoint.

## Comparison to Other Architectures

AetherNet positions itself uniquely in the landscape of SISR models:

-   **vs. ESRGAN:** ESRGAN is a legendary baseline for perceptual quality, known for its ability to generate fine details. However, it is computationally heavy. AetherNet offers a much faster and more efficient alternative, making it suitable for real-time applications where ESRGAN would be too slow. AetherNet's INT8 models, in particular, are orders of magnitude faster.

-   **vs. SPAN / RealPLKSR:** These models also explore advanced architectural designs like spatial attention and large kernels to achieve high quality. AetherNet shares the use of large kernels but combines it with structural reparameterization and a first-class commitment to QAT. This gives it an edge in achieving the absolute best *performance-per-watt* on deployed hardware, especially when targeting INT8 inference.

-   **Uniqueness:** AetherNet's main differentiator is its holistic focus on **deployable performance**. The architecture and the release pipeline were co-designed to produce models that are not just high-quality but also practical, efficient, and easy to integrate into real-world applications using runtimes like ONNX Runtime, TensorRT, and DirectML.

## Network Presets

AetherNet comes in several sizes to fit different needs, from mobile devices to high-end GPUs:

-   `aether_tiny`: For real-time applications where speed is the absolute priority.
-   `aether_small`: A balanced choice offering a great mix of speed and quality.
-   `aether_medium`: A higher-quality option for less constrained environments.
-   `aether_large`: The highest-quality variant for when visual fidelity is paramount.

## The Deployment Pipeline: `aether_release.py`

A training checkpoint is not a deployment artifact. The release script is a crucial tool that bridges this gap.

### Goal
To take a single QAT-trained `.pth` checkpoint and automatically generate a full suite of validated, optimized models for every common use case.

### How to Use
The script is designed to be simple to run.

```bash
# Activate your python environment
# pip install -r requirements.txt (make sure torch, onnx, onnxruntime, etc. are installed)

python3 aether_release.py \
    --model-path /path/to/your/net_g_final.pth \
    --output-dir /path/to/your/release_folder \
    --validation-dir /path/to/a/folder/of/images
```

-   `--model-path`: Path to the final, finetuned QAT checkpoint.
-   `--output-dir`: Where the final models will be saved.
-   `--validation-dir`: A folder with a few sample images. The script will use one to perform a "smoke test" on each created model to ensure it runs without errors.

### Output Files
The script will generate up to six files in the output directory:
-   **PyTorch Models:**
    -   `model_fp32.pth`: A fused, float32 model.
    -   `model_fp16.pth`: A fused, half-precision model for NVIDIA GPUs.
    -   `model_int8.pth`: A fused, quantized INT8 model for use within PyTorch.
-   **ONNX Models:**
    -   `model_fp32.onnx`: The most compatible format for general use.
    -   `model_fp16.onnx`: Ideal for deployment on NVIDIA GPUs via TensorRT or ONNX Runtime.
    -   `model_int8.onnx`: **The fastest format.** Ideal for CPU inference and for deployment on NVIDIA/DirectML backends that support INT8.

All ONNX models are saved with dynamic input axes and include metadata that allows tools like [Spandrel](https://github.com/chaiNNer-org/spandrel) and [ChaiNNer](https://chainner.app/) to load them automatically.

## License
AetherNet is released under the **MIT License**. See the `LICENSE` file for more details.

## Created By
AetherNet was created by **Philip Hofmann** in a collaborative effort with advanced AI assistance. This project represents a fusion of human architectural design and AI-driven debugging and code generation, showcasing a modern workflow for creating complex deep learning systems.
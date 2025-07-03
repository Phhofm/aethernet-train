# AetherNet-Train: Training Framework for AetherNet SISR

This repository provides the official training and deployment pipeline for the **AetherNet** Single-Image Super-Resolution (SISR) models. It is built upon the excellent **[neosr](https://github.com/neosr-project/neosr)** framework, which has been adapted specifically for training and optimizing AetherNet.

The primary goal of this project is to offer a complete, end-to-end workflow for creating highly efficient, production-ready super-resolution models that leverage modern hardware capabilities.

-   **This Repository:** [https://github.com/Phhofm/aethernet-train](https://github.com/Phhofm/aethernet-train)
-   **AetherNet Architecture:** [https://github.com/Phhofm/aethernet](https://github.com/Phhofm/aethernet)
-   **Original Neosr Framework:** [https://github.com/neosr-project/neosr](https://github.com/neosr-project/neosr)

> **Note:** While this framework is optimized for AetherNet, it is based on the powerful `neosr` project. If you wish to train other architectures (HAT, SwinIR, etc.), please refer to the official [neosr repository](https://github.com/neosr-project/neosr), which is actively maintained and supports a wider range of models.

## The AetherNet Philosophy: Speed Without Compromise

AetherNet was designed from the ground up with one primary goal: **maximum inference speed** on a wide range of hardware, from high-end GPUs to CPUs. The architecture achieves this without a significant loss in quality by focusing on two key technologies:

1.  **Quantization-Aware Training (QAT):** The ultimate goal of this pipeline is to produce an **INT8 ONNX model**. Integer-based (INT8) arithmetic is significantly faster than floating-point (FP32) arithmetic on modern hardware. QAT involves fine-tuning the model to be aware of quantization's constraints, which minimizes the quality degradation that typically occurs during conversion. This results in models that are nearly identical in quality to their FP32 counterparts but are orders of magnitude faster at inference time.

2.  **Structural Reparameterization:** During training, AetherNet uses complex blocks with multiple convolution branches to increase its learning capacity. Before deployment, these complex blocks are mathematically **fused** into single, simple convolution layers. This gives you the quality benefits of a more complex architecture with the inference speed of a much simpler one—the "best of both worlds."

## Installation and Setup

To get started, clone this repository and set up the Python environment using the provided `requirements.txt`.

```bash
# 1. Clone the repository
git clone https://github.com/Phhofm/aethernet-train.git
cd aethernet-train

# 2. Create a Python virtual environment
python3 -m venv venv

# 3. Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 4. Install the required dependencies
pip install -r requirements.txt
```

You are now ready to train and deploy AetherNet models.

## The Complete Workflow: From Training to Deployment

The recommended workflow involves a two-stage training process followed by a release step. This ensures you get both a maximum-quality FP32 model and a maximum-speed INT8 model.

### Stage 1: FP32 Pre-training (For Quality)

The first step is to train a standard, high-quality FP32 model. This model will serve as an excellent starting point (a "pre-train") for the QAT phase.

1.  **Configure:** Use one of the training configurations in the `configs/` directory, such as `2xaether_tiny.toml`. Ensure that `enable_qat` is set to `false` (or commented out) and that `ema` and `grad_clip` are enabled for stability and best results.

    ```toml
    # In configs/2xaether_tiny.toml
    name = "2xaether_tiny_fp32"
    model_type = "image"
    ...
    [network_g]
    type = "aether_tiny"

    [train]
    enable_qat = false
    ema = 0.999
    grad_clip = true
    ...
    ```

2.  **Train:** Start the training from the `neosr` directory.

    ```bash
    cd neosr
    python3 train.py --opt ../configs/2xaether_tiny.toml
    ```

3.  **Result:** Let this run until the validation metrics plateau (e.g., 200,000+ iterations). The final saved checkpoint in the `experiments/` folder is your high-quality FP32 model.

### Stage 2: QAT Fine-tuning (For Speed)

Now, use the model from Stage 1 to fine-tune a new model that is resilient to quantization.

1.  **Configure:** Create a new configuration file (e.g., `2xaether_tiny_qat.toml`).
2.  **Modify the config:**
    *   Set a new `name` for the experiment.
    *   In the `[path]` section, set `pretrain_network_g` to the path of your best model from Stage 1 (e.g., `../experiments/2xaether_tiny_fp32/models/net_g_200000.pth`).
    *   Set `strict_load_g = false` to allow loading the weights into the slightly different QAT architecture.
    *   In the `[train]` section, set `enable_qat = true`.
    *   **Important:** Use a lower learning rate for fine-tuning (e.g., `lr = 2e-5`).

3.  **Train:** Start the new training run. This phase is much shorter, typically requiring only 50k-100k iterations to adapt the weights.

### Stage 3: The Release Pipeline

The `aether_release.py` script is used to convert your trained checkpoints into deployable formats. It correctly handles both QAT and non-QAT models.

1.  **Navigate to the `archs` directory:**
    ```bash
    cd neosr/neosr/archs
    ```

2.  **Run the script:**
    ```bash
    python3 aether_release.py \
        --model-path /path/to/your/net_g_final.pth \
        --output-dir /path/to/your/release_folder \
        --validation-dir /path/to/a/folder/of/images
    ```
    *   `--model-path`: Path to the checkpoint you want to convert (either the FP32 one or the QAT one).
    *   `--output-dir`: Where the final models will be saved.
    *   `--validation-dir`: A folder with sample images for a quick sanity check.

3.  **Outputs:**
    *   If you run it on a **non-QAT model**, it will produce fused `_fp32.pth`, `_fp16.pth`, and their corresponding `.onnx` versions.
    *   If you run it on a **QAT model**, it will produce all of the above, plus the crucial `_int8.onnx` file, ready for high-performance inference with ONNX Runtime, TensorRT, or DirectML.

## Pre-trained Models

*(This section is a placeholder. You can add your GitHub Releases links here later.)*

Pre-trained models for each AetherNet variant will be made available here. For each variant (`tiny`, `small`, `medium`, `large`), two models will be provided:

-   **FP32 Model:** The highest-quality model, trained without QAT. Ideal for users with powerful GPUs who prioritize maximum visual fidelity.
-   **QAT Model:** The model fine-tuned for quantization. Use this checkpoint with the release script to generate the super-fast INT8 ONNX file.

## License

This project is released under the **Apache 2.0 License** (because neosr is under that license)

This work is built upon the `neosr` framework. We extend our sincere gratitude to the original `neosr` authors and community for their foundational work.
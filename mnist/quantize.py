"""Quantize trained MNIST MLP to int8 and export weights/test data as .npy files.

Per-layer symmetric quantization:
  scale = max(|W|) / 127
  W_int8 = clamp(round(W / scale), -128, 127)

For input quantization, we quantize the normalized input to int8.

Export format:
  mnist/export/layer{1,2,3}_weights.npy  (int8)
  mnist/export/layer{1,2,3}_bias.npy     (int32)
  mnist/export/layer{1,2,3}_scale.npy    (float32 — weight_scale)
  mnist/export/input_scale.npy           (float32 — input quantization scale)
  mnist/export/test_images.npy           (int8, shape [10, 784])
  mnist/export/test_labels.npy           (int64, shape [10])
"""

import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from train import MnistMLP


def quantize_tensor_symmetric(tensor, num_bits=8):
    """Symmetric per-tensor quantization to int8."""
    max_val = tensor.abs().max().item()
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return quantized, scale


def quantize_and_export(model_path=None, export_dir=None):
    if model_path is None:
        model_path = Path(__file__).parent / "model.pth"
    if export_dir is None:
        export_dir = Path(__file__).parent / "export"
    export_dir.mkdir(exist_ok=True)

    model = MnistMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    layers = [model.fc1, model.fc2, model.fc3]

    # Quantize weights
    for i, layer in enumerate(layers, 1):
        w_int8, w_scale = quantize_tensor_symmetric(layer.weight.data)
        # Quantize bias to int32 (bias is added to int32 accumulators)
        # bias_int32 = round(bias / (input_scale * weight_scale))
        # But we store raw bias scaled by weight_scale for now;
        # the inference driver will handle full scaling
        b_float = layer.bias.data
        # Store bias as float32 for flexibility — the driver handles scaling
        np.save(export_dir / f"layer{i}_weights.npy", w_int8.numpy())
        np.save(export_dir / f"layer{i}_bias.npy", b_float.numpy().astype(np.float32))
        np.save(export_dir / f"layer{i}_scale.npy", np.array([w_scale], dtype=np.float32))
        print(f"Layer {i}: weight shape {tuple(w_int8.shape)}, scale {w_scale:.6f}")

    # Quantize input: use the normalization range to determine input scale
    # MNIST normalized: (x - 0.1307) / 0.3081, range roughly [-0.42, 2.82]
    # We'll quantize the normalized input symmetrically
    # First find the actual range from test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(
        str(Path(__file__).parent / "data"), train=False, transform=transform
    )

    # Find max absolute value across a sample
    max_abs = 0
    for i in range(min(1000, len(test_dataset))):
        img, _ = test_dataset[i]
        max_abs = max(max_abs, img.abs().max().item())

    input_scale = max_abs / 127.0
    np.save(export_dir / "input_scale.npy", np.array([input_scale], dtype=np.float32))
    print(f"Input scale: {input_scale:.6f} (max_abs: {max_abs:.4f})")

    # Export 10 test images (one per digit class)
    test_images = []
    test_labels = []
    found_digits = set()

    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        if label not in found_digits:
            found_digits.add(label)
            img_flat = img.view(784)
            img_int8 = torch.clamp(torch.round(img_flat / input_scale), -128, 127).to(torch.int8)
            test_images.append(img_int8.numpy())
            test_labels.append(label)
        if len(found_digits) == 10:
            break

    # Sort by label
    order = sorted(range(10), key=lambda i: test_labels[i])
    test_images = np.stack([test_images[i] for i in order])
    test_labels = np.array([test_labels[i] for i in order], dtype=np.int64)

    np.save(export_dir / "test_images.npy", test_images)
    np.save(export_dir / "test_labels.npy", test_labels)
    print(f"Exported {len(test_labels)} test images, labels: {test_labels.tolist()}")

    return export_dir


def verify_quantized_accuracy(export_dir=None, model_path=None):
    """Run quantized inference in numpy to verify accuracy."""
    if export_dir is None:
        export_dir = Path(__file__).parent / "export"
    if model_path is None:
        model_path = Path(__file__).parent / "model.pth"

    # Load original model for reference
    model = MnistMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Load quantized weights
    weights = []
    biases = []
    w_scales = []
    for i in range(1, 4):
        weights.append(np.load(export_dir / f"layer{i}_weights.npy"))
        biases.append(np.load(export_dir / f"layer{i}_bias.npy"))
        w_scales.append(np.load(export_dir / f"layer{i}_scale.npy")[0])

    input_scale = np.load(export_dir / "input_scale.npy")[0]

    # Run quantized inference on full test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(
        str(Path(__file__).parent / "data"), train=False, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    correct_quant = 0
    correct_float = 0
    total = 0

    for data, target in test_loader:
        # Float reference
        with torch.no_grad():
            float_pred = model(data).argmax(dim=1)
            correct_float += (float_pred == target).sum().item()

        # Quantized inference (numpy, float accumulation for simplicity)
        x_np = data.view(-1, 784).numpy()

        for img_idx in range(x_np.shape[0]):
            x = x_np[img_idx]

            # Quantize input
            x_int8 = np.clip(np.round(x / input_scale), -128, 127).astype(np.int8)
            current_scale = input_scale

            for layer_idx in range(3):
                w = weights[layer_idx].astype(np.int32)
                x_i32 = x_int8.astype(np.int32)

                # Integer matmul
                acc = x_i32 @ w.T  # shape: (out_features,)

                # Dequantize accumulator and add float bias
                acc_float = acc.astype(np.float64) * current_scale * w_scales[layer_idx]
                acc_float += biases[layer_idx].astype(np.float64)

                if layer_idx < 2:  # ReLU for first two layers
                    acc_float = np.maximum(acc_float, 0)

                # Requantize to int8 for next layer
                if layer_idx < 2:
                    out_max = np.abs(acc_float).max()
                    if out_max == 0:
                        current_scale = 1.0
                    else:
                        current_scale = out_max / 127.0
                    x_int8 = np.clip(np.round(acc_float / current_scale), -128, 127).astype(np.int8)
                else:
                    # Last layer: just use float logits for argmax
                    pred = np.argmax(acc_float)

            if pred == target[img_idx].item():
                correct_quant += 1
            total += 1

    float_acc = correct_float / total
    quant_acc = correct_quant / total
    print(f"Float accuracy: {float_acc:.4f}")
    print(f"Quantized accuracy: {quant_acc:.4f}")
    return quant_acc, float_acc


if __name__ == "__main__":
    quantize_and_export()
    verify_quantized_accuracy()

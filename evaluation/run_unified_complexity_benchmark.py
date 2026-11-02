#!/usr/bin/env python3
"""
Step 2f: Unified Model Complexity & Efficiency Benchmark on GPU
================================================================
Measures all 7 models + nnU-Net under unified conditions:
  - Hardware: GPU (NVIDIA A100 or available accelerator)
  - ROI Size: 128 x 128 x 64
  - Precision: bfloat16 AMP
  - Sliding Window Overlap: 0.5
  - Latency: 10 warm-up runs + 30 timed repeats (median + IQR in ms)
  - Volume Size: [1, C, 384, 384, 128]
  - Metrics:
    1. Parameters (M)
    2. FLOPs per patch (G) via PyTorch FlopCounterMode
    3. Inference Latency per volume: Median and IQR (ms)
    4. Peak GPU VRAM (MB)
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from monai.inferers import sliding_window_inference

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.attention_isolated_model import get_isolated_model


def get_nnunet_model(ckpt_path=None):
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    arch_kwargs = {
        'n_stages': 6,
        'features_per_stage': [32, 64, 128, 256, 320, 320],
        'conv_op': torch.nn.modules.conv.Conv3d,
        'kernel_sizes': [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        'strides': [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
        'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
        'conv_bias': True,
        'norm_op': torch.nn.modules.instancenorm.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': torch.nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True}
    }
    net = PlainConvUNet(input_channels=3, num_classes=2, **arch_kwargs)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["network_weights"])
    return net


def benchmark_model(model_name: str, model: nn.Module, in_channels: int, device: torch.device, roi_size=(128, 128, 64), vol_shape=(384, 384, 128)):
    print(f"\n>>> Benchmarking {model_name} (in_channels={in_channels}) on {device}...")
    model = model.to(device).eval()

    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    patch_tensor = torch.randn(1, in_channels, *roi_size, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with FlopCounterMode(display=False) as fc:
            _ = model(patch_tensor)
        flops_patch_g = fc.get_total_flops() / 1e9

    vol_tensor = torch.randn(1, in_channels, *vol_shape, device=device, dtype=torch.bfloat16)

    print("    Running 10 warmup runs...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10):
            _ = sliding_window_inference(vol_tensor, roi_size, sw_batch_size=8, predictor=model, overlap=0.5)
            torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)

    print("    Running 30 timed measurement runs...")
    latencies_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(30):
            start_event.record()
            _ = sliding_window_inference(vol_tensor, roi_size, sw_batch_size=8, predictor=model, overlap=0.5)
            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))

    lat_arr = np.array(latencies_ms)
    med_lat_ms = float(np.median(lat_arr))
    iqr_lat_ms = float(np.percentile(lat_arr, 75) - np.percentile(lat_arr, 25))
    mean_lat_ms = float(np.mean(lat_arr))
    std_lat_ms = float(np.std(lat_arr))

    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    result = {
        "model": model_name,
        "in_channels": in_channels,
        "parameters_m": round(n_params_m, 2),
        "flops_per_patch_g": round(flops_patch_g, 2),
        "latency_median_ms": round(med_lat_ms, 2),
        "latency_iqr_ms": round(iqr_lat_ms, 2),
        "latency_mean_ms": round(mean_lat_ms, 2),
        "latency_std_ms": round(std_lat_ms, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }

    print(f"    Params: {result['parameters_m']:.2f} M | FLOPs: {result['flops_per_patch_g']:.2f} G")
    print(f"    Latency: {result['latency_median_ms']:.1f} ms (IQR: {result['latency_iqr_ms']:.1f} ms) | VRAM: {result['peak_vram_mb']:.1f} MB")

    del model, patch_tensor, vol_tensor
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Model Complexity and Efficiency Benchmark")
    parser.add_argument("--out-dir", type=str, default="./results")
    parser.add_argument("--nnunet-ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("  STEP 2F: UNIFIED COMPLEXITY BENCHMARK")
    print(f"  Device: {device} ({gpu_name})")
    print("==========================================================================")

    models_to_bench = [
        ("standard_scse", lambda: get_isolated_model("standard", in_channels=5, out_channels=1), 5),
        ("pointwise_deep", lambda: get_isolated_model("pointwise_deep", in_channels=5, out_channels=1), 5),
        ("matched_3x3x3", lambda: get_isolated_model("boundary_matched", in_channels=5, out_channels=1), 5),
        ("full_3x3x3", lambda: get_isolated_model("boundary_full", in_channels=5, out_channels=1), 5),
        ("segresnet_5ch", lambda: get_isolated_model("segresnet", in_channels=5, out_channels=1), 5),
        ("no_scse", lambda: get_isolated_model("no_scse", in_channels=5, out_channels=1), 5),
        ("kg_scse", lambda: get_isolated_model("kg_scse", in_channels=5, out_channels=1), 5),
        ("nnunet_3d_fullres", lambda: get_nnunet_model(args.nnunet_ckpt), 3),
    ]

    all_results = {}
    records = []

    for name, init_fn, in_ch in models_to_bench:
        try:
            model = init_fn()
            res = benchmark_model(name, model, in_ch, device)
            all_results[name] = res
            records.append(res)
        except Exception as e:
            print(f"    Skipping {name}: {e}")

    summary = {
        "metadata": {
            "device": gpu_name,
            "precision": "bfloat16 AMP"
        },
        "benchmark_results": all_results
    }

    with open(out_dir / "model_complexity_benchmark.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_dir / "model_complexity_benchmark.csv", index=False)
    print(f"\n[done] Saved results to {out_dir}")


if __name__ == "__main__":
    main()

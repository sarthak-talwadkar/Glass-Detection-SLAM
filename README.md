# Glass-Detection-SLAM

**Real-time glass surface detection integrated into a visual SLAM pipeline**, enabling accurate occupancy mapping in environments with transparent obstacles. Core contribution: a multi-stage inference optimization pipeline that brings GDNet from **6–8 FPS → 30+ FPS** on embedded hardware without retraining.

---

## Demo

![Glass Detection SLAM — live inference demo](assets/demo.gif)

> GDNet running at 30+ FPS with temporal filtering. Glass surfaces (highlighted overlay) are detected in real time and injected into the SLAM occupancy map as solid obstacles.

---

## Table of Contents

- [Motivation](#motivation)
- [System Architecture](#system-architecture)
- [Optimization Pipeline](#optimization-pipeline)
  - [1. TorchScript Export](#1-torchscript-export)
  - [2. ONNX Graph Optimization](#2-onnx-graph-optimization)
  - [3. TensorRT FP16 Quantization](#3-tensorrt-fp16-quantization)
  - [4. TensorRT INT8 Calibration](#4-tensorrt-int8-calibration)
  - [5. Temporal Filtering](#5-temporal-filtering)
- [SLAM Integration](#slam-integration)
- [Performance Benchmarks](#performance-benchmarks)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Motivation

Glass surfaces are a fundamental failure mode for robotic perception systems:

- **LiDAR** transmits through glass, producing no return — glass walls appear as free space
- **RGB-D cameras** (structured light / ToF) receive corrupted or absent depth at transparent surfaces
- **Standard SLAM** (ORB-SLAM3, gmapping) has no semantic understanding of transparency — it treats missing depth as navigable space

The result: robots collide with glass walls, and occupancy maps have holes where solid obstacles exist. This project adds a learned glass detection front-end (GDNet) to the SLAM pipeline, projecting glass masks into the occupancy map as solid obstacles — without modifying the SLAM backend.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Input Stream                            │
│              (RGB frames @ 30Hz — camera / ROS bag)            │
└───────────────────────────┬────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │     Preprocessing          │
              │  Resize → 416×416          │
              │  Normalize (ImageNet μ,σ)  │
              │  NCHW layout, FP16 cast    │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       GDNet Inference      │
              │   TensorRT INT8 Engine     │
              │   (ResNet backbone +       │
              │    multi-scale context     │
              │    feature module)         │
              └─────────────┬──────────────┘
                            │  Raw glass probability map
              ┌─────────────▼──────────────┐
              │    Post-processing         │
              │  Sigmoid → binary mask     │
              │  Threshold @ 0.45          │
              │  Morphological close (3×3) │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │    Temporal Filter         │
              │  Ring buffer (N=5 frames)  │
              │  Pixel-wise majority vote  │
              │  Reduces flicker & FP      │
              └─────────────┬──────────────┘
                            │  Stable glass mask
        ┌───────────────────┴──────────────────────┐
        │                                          │
┌───────▼────────┐                    ┌────────────▼──────────┐
│  Depth Masking │                    │   Map Augmentation    │
│ Zero out depth │                    │ Project mask → world  │
│ at glass pixels│                    │ Mark as occupied cells│
│ before SLAM    │                    │ in occupancy grid     │
│ front-end      │                    └───────────────────────┘
└───────┬────────┘
        │
┌───────▼────────┐
│  SLAM Backend  │
│ (ORB-SLAM3 /  │
│  gmapping)    │
│ Localization + │
│ Mapping        │
└────────────────┘
```

---

## Optimization Pipeline

GDNet in its original PyTorch FP32 form runs at **6–8 FPS** — far below the 30Hz camera rate needed for real-time SLAM. The following staged optimization pipeline was applied to close that gap.

---

### 1. TorchScript Export

**Why:** Eliminates Python interpreter overhead and enables runtime deployment without a PyTorch installation.

```python
import torch
from gdnet.model import GDNet

model = GDNet()
model.load_state_dict(torch.load("checkpoints/gdnet.pth"))
model.eval()

# Trace with representative input
example_input = torch.rand(1, 3, 416, 416).cuda()
traced = torch.jit.trace(model, example_input)
torch.jit.save(traced, "checkpoints/gdnet_scripted.pt")
```

**What it does internally:**
- Freezes the computation graph, eliminating dynamic dispatch per operation
- Fuses eligible elementwise ops (e.g., BatchNorm into Conv2d via `torch.jit.optimize_for_inference`)
- Removes autograd tracking overhead

**Speedup:** ~6–8 FPS → ~12–15 FPS

---

### 2. ONNX Graph Optimization

**Why:** ONNX provides a hardware-agnostic intermediate representation that TensorRT (and other backends) consume. Exporting through ONNX also allows graph-level optimization passes before engine compilation.

```python
torch.onnx.export(
    model,
    example_input,
    "checkpoints/gdnet.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["glass_map"],
    dynamic_axes={"input": {0: "batch_size"}},
    do_constant_folding=True,   # folds constant subgraphs at export time
)
```

**Post-export graph optimization with `onnxoptimizer`:**

```python
import onnx
from onnxoptimizer import optimize

model_onnx = onnx.load("checkpoints/gdnet.onnx")
passes = [
    "eliminate_identity",
    "eliminate_unused_initializer",
    "fuse_consecutive_transposes",
    "fuse_transpose_into_gemm",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_bn_into_conv",          # critical: absorbs BN params into Conv weights
]
optimized = optimize(model_onnx, passes)
onnx.save(optimized, "checkpoints/gdnet_optimized.onnx")
```

**Key fusion — BN into Conv:** BatchNorm can be algebraically absorbed into the preceding Conv layer's weights and biases at inference time, eliminating a separate memory read/write pass per feature map. For ResNet-50 backbone, this removes ~16 BN operations from the critical path.

---

### 3. TensorRT FP16 Quantization

**Why:** FP16 halves memory bandwidth requirements and uses the GPU's Tensor Cores (on Jetson / RTX GPUs), typically yielding 1.5–2× speedup over FP32 with negligible accuracy loss for detection tasks.

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("checkpoints/gdnet_optimized.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
config.set_flag(trt.BuilderFlag.FP16)   # enable FP16 Tensor Core paths

engine = builder.build_serialized_network(network, config)
with open("checkpoints/gdnet_fp16.engine", "wb") as f:
    f.write(engine)
```

**What TensorRT does under the hood:**
- **Layer fusion:** Consecutive Conv + BN + ReLU fused into a single CUDA kernel call (CudnnConvBiasActivation)
- **Kernel auto-tuning:** Benchmarks multiple CUDA kernel implementations per layer and selects the fastest for the specific GPU
- **Memory layout optimization:** Converts feature maps to NHWC or NC/32HW32 (Tensor Core-friendly) layout automatically
- **Reformatting minimization:** Inserts format conversion layers only where necessary to minimize layout-change overhead

**Speedup:** ~15 FPS → ~22–25 FPS

---

### 4. TensorRT INT8 Post-Training Quantization (PTQ)

**Why:** INT8 arithmetic is 2–4× faster than FP16 on Tensor Cores and halves activation memory. The challenge is calibrating per-layer quantization scales so that the reduced dynamic range doesn't hurt detection quality.

**Calibration dataset:** 500 representative frames covering varied glass/no-glass scenes are used to determine per-tensor activation ranges via histogram analysis.

```python
class GDNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_images, cache_file="calib.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.data = self._load_and_preprocess(calib_images)  # N×3×416×416 FP32
        self.index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes)

    def get_batch(self, names):
        if self.index >= len(self.data):
            return None
        cuda.memcpy_htod(self.device_input, self.data[self.index])
        self.index += 1
        return [int(self.device_input)]

    def get_batch_size(self):
        return 1

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Build INT8 engine
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = GDNetCalibrator(calib_images)
```

**Calibration algorithm — Entropy Calibration (IInt8EntropyCalibrator2):**
- Collects activation histograms over the calibration set for each tensor
- Finds the threshold T that minimizes KL divergence between the original FP32 distribution and the quantized INT8 representation
- Stores per-layer scale factors in the calibration cache for reproducible builds

**Selective precision (mixed precision):** Layers sensitive to quantization error (typically the final detection head) are kept in FP16:

```python
# Force final output layer to FP16 to preserve detection confidence accuracy
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if "output" in layer.name:
        layer.precision = trt.DataType.HALF
        layer.set_output_type(0, trt.DataType.HALF)
```

**Speedup:** ~25 FPS → **30+ FPS**

---

### 5. Temporal Filtering

**Why:** Frame-by-frame detection produces flickering masks — a glass pixel detected in frame N may be absent in frame N+1 due to slight viewpoint shift or lighting change. This causes noisy occupancy map updates and instability in downstream planning.

**Implementation — pixel-wise majority vote over a ring buffer:**

```python
from collections import deque
import numpy as np

class TemporalFilter:
    def __init__(self, window=5, threshold=0.5):
        self.buffer = deque(maxlen=window)
        self.threshold = threshold  # fraction of frames that must agree

    def update(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: H×W binary array (0/1 float)
        returns: temporally smoothed H×W binary mask
        """
        self.buffer.append(mask.astype(np.float32))
        stacked = np.stack(self.buffer, axis=0)          # T×H×W
        mean_activation = stacked.mean(axis=0)            # H×W in [0,1]
        return (mean_activation >= self.threshold).astype(np.uint8)
```

**Effect:**
- A pixel is marked as glass only if it was detected as glass in ≥50% of the last 5 frames
- Eliminates single-frame false positives caused by specular highlights or motion blur
- Introduces a maximum latency of ~5 frames (167ms @ 30Hz) for new glass to appear in the map — acceptable for static glass surfaces

**Morphological post-processing** applied before temporal filter to close small holes in the raw detection mask:

```python
import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
```

---

## SLAM Integration

The stabilized glass mask is integrated into the SLAM pipeline via **two parallel channels**:

### Channel 1 — Depth Masking (upstream intervention)

Before the RGB-D frame reaches the SLAM front-end, glass pixels are zeroed out in the depth image:

```python
def mask_depth(depth_frame: np.ndarray, glass_mask: np.ndarray) -> np.ndarray:
    masked = depth_frame.copy()
    masked[glass_mask == 1] = 0   # treat glass as missing depth
    return masked
```

This prevents the SLAM system from creating false features or erroneous 3D points at glass surfaces.

### Channel 2 — Occupancy Map Augmentation (downstream injection)

The glass mask is back-projected into 3D world coordinates using the camera intrinsics and current pose estimate from SLAM, then written as occupied cells into the occupancy grid:

```python
def project_glass_to_map(mask, depth, K, T_world_cam, occ_map, resolution):
    """
    mask:        H×W binary glass mask
    depth:       H×W depth (may be zero at glass — use estimated depth or fixed range)
    K:           3×3 camera intrinsic matrix
    T_world_cam: 4×4 SE(3) pose from SLAM
    occ_map:     occupancy grid object
    resolution:  meters per cell
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    ys, xs = np.where(mask == 1)

    # Use fixed depth estimate for glass (since actual depth is missing)
    z = np.full(len(xs), fill_value=GLASS_DEPTH_ESTIMATE)

    # Backproject to camera frame
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)  # N×4

    # Transform to world frame
    pts_world = (T_world_cam @ pts_cam.T).T[:, :3]

    # Mark as occupied in map
    for pt in pts_world:
        occ_map.set_occupied(pt, radius=resolution)
```

---

## Performance Benchmarks

| Stage | Method | FPS | Latency (ms) |
|---|---|---|---|
| Baseline | PyTorch FP32 | 6–8 | ~140 |
| Stage 1 | TorchScript | ~12–15 | ~70 |
| Stage 2 | ONNX + graph opts | ~16–18 | ~58 |
| Stage 3 | TensorRT FP16 | ~22–25 | ~42 |
| Stage 4 | TensorRT INT8 (PTQ) | **30+** | **~30** |
| + Filter | INT8 + temporal filter | 30+ | ~30 + 5-frame buffer |

> Benchmarked on NVIDIA Jetson [model], input resolution 416×416, batch size 1.

**~4.3× end-to-end speedup** from baseline to final optimized pipeline.

---

## Installation

```bash
git clone https://github.com/sarthak-talwadkar/Glass-Detection-SLAM.git
cd Glass-Detection-SLAM
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch ≥ 1.10, TensorRT ≥ 8.x, ONNX ≥ 1.12, onnxoptimizer, OpenCV ≥ 4.5, CUDA ≥ 11.x, pycuda

### Build TensorRT INT8 Engine

```bash
# Place ~500 calibration images in data/calibration/
python export/build_trt_engine.py \
    --weights checkpoints/gdnet.pth \
    --onnx checkpoints/gdnet_optimized.onnx \
    --precision int8 \
    --calib-dir data/calibration/ \
    --output checkpoints/gdnet_int8.engine
```

---

## Usage

### Detection demo (no SLAM)

```bash
python demo.py \
    --input data/sample.mp4 \
    --engine checkpoints/gdnet_int8.engine \
    --temporal-window 5 \
    --threshold 0.45
```

### Full pipeline with SLAM

```bash
python run_slam.py \
    --input data/input_sequence/ \
    --engine checkpoints/gdnet_int8.engine \
    --slam orbslam3 \
    --vocab checkpoints/ORBvoc.txt \
    --camera-config config/camera.yaml \
    --temporal-window 5
```

---

## Project Structure

```
Glass-Detection-SLAM/
├── gdnet/
│   ├── model.py            # GDNet architecture (ResNet + MCFM)
│   └── checkpoints/        # Pretrained weights
├── export/
│   ├── export_onnx.py      # PyTorch → ONNX export
│   ├── build_trt_engine.py # ONNX → TensorRT INT8/FP16 engine builder
│   └── calibrator.py       # IInt8EntropyCalibrator2 implementation
├── slam/
│   ├── integration.py      # Depth masking + map augmentation
│   └── occupancy.py        # Occupancy grid utilities
├── utils/
│   ├── temporal_filter.py  # Ring buffer majority-vote filter
│   └── visualize.py        # Overlay visualization tools
├── demo.py                 # Standalone glass detection demo
├── run_slam.py             # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## References

- Mei, H., et al. *"Don't Hit Me! Glass Detection in Real-world Scenes."* CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mei_Dont_Hit_Me_Glass_Detection_in_Real-World_Scenes_CVPR_2020_paper.pdf)
- NVIDIA TensorRT Developer Guide — INT8 Calibration. [[Docs]](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- Mur-Artal, R. et al. *"ORB-SLAM3."* IEEE T-RO 2021. [[Paper]](https://arxiv.org/abs/2007.11898)

---

## Author

**Sarthak Talwadkar**  
MS Robotics, Northeastern University  
[LinkedIn](https://linkedin.com/in/sarthak-talwadkar) · [GitHub](https://github.com/sarthak-talwadkar)

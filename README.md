# Edge ML: Keyword Detection & Image Classification on Arduino Nano 33 BLE

---

## Project 1 — Keyword Detection

Keyword spotting system that classifies spoken city names (Bangalore, Pokhara) plus noise and unknown, with real-time on-device inference using the board's onboard PDM microphone.

Built on Edge Impulse: data collection → MFCC feature extraction → 1D CNN training → int8 quantization → TFLite Micro deployment.

### Dataset

~1h 12min of audio across four classes. Bangalore and Pokhara samples recorded via Arduino microphone and smartphones, with variation across accents (Finnish, English, Chinese, French, German, Indian, Nepali, Italian), genders, and speech speeds (0.75x–1.5x). Synthetic speech from text-to-speech tools added for coverage. Noise and unknown sourced from a public Edge Impulse keyword spotting dataset. 85/15 train/test split.

### Model

- Gaussian noise layer at input for robustness
- Two 1D conv layers (16 and 32 filters) with max pooling and dropout
- Dense layer (64 neurons), softmax output
- SpecAugment augmentation, Adam optimizer, LR reduction on val loss plateau
- 50 epochs, batch size 32

Window size: 4096ms gave best accuracy but caused out-of-memory errors on-device. Reduced to **2000ms / 500ms stride** for a workable balance between accuracy and deployment.

Feature extraction: MFCC (26 coefficients, 32 filters) chosen over MFE. MFE hit 62.5% overall accuracy vs 57.2% for MFCC, but MFCC gave more balanced F1 across the target keyword classes and required less RAM.

### Results

| Metric | Value |
|--------|-------|
| Validation accuracy | 70.0% |
| Test accuracy | 57.2% |
| Validation AUC | 0.90 |
| Quantization | int8 |
| MFCC latency | 342 ms |
| Total latency | 355 ms |
| RAM | 30.8 KB |
| Flash | 82.4 KB |

Distribution shift evaluation (Azerbaijani speakers, not in training set): accuracy dropped to 40.8%, with most predictions falling into *uncertain* — the model is sensitive to unseen pronunciation patterns despite training on multiple accent groups.

---

## Project 2 — Image Classification (Rock Paper Scissors)

End-to-end image classification pipeline for hand gesture recognition (paper / rock / scissors) with inference running on-device from a live OV7670 camera feed.

Built in Python on Colab (TF 2.20, TFMOT) → TFLite Micro on Arduino → OV7670 integration → host-side verification over serial.

### Model

Spatial-aware CNN with 64×64 grayscale input. Strided convolutions reduce spatial dimensions early; a Flatten layer (not GlobalAveragePooling) preserves left/right/top/bottom hand-position information into the dense head. GAP collapsed each feature map to a single value and caused the model to predict based on average brightness rather than hand shape, which biased everything toward "paper".

| Layer | Output | Params |
|-------|--------|--------|
| Conv2D(8, 3×3, stride 2) + ReLU | 32×32×8 | 80 |
| MaxPool(2×2) | 16×16×8 | 0 |
| Conv2D(16, 3×3, stride 2) + ReLU | 8×8×16 | 1,168 |
| Flatten | 1,024 | 0 |
| Dense(16) + ReLU + Dropout(0.2) | 16 | 16,400 |
| Dense(3) + Softmax | 3 | 51 |

**17,699 total parameters.**

### Compression

Six pruning configurations (whole-model and conv-only at sparsities 0.3/0.5/0.7) and three quantization strategies evaluated against the float32 baseline.

Key findings:
- Pruning alone produces no on-disk size reduction — the TFLite flatbuffer still stores zeroed weights at full float32 precision. It acts as regularisation: whole-model pruning at s=0.30 improved test accuracy by +3.2pp over baseline.
- Quantization drives the actual size reduction: all three methods cut ~70% with negligible accuracy drop.
- Only full INT8 (Task 2C) is deployable on the Arduino Cortex-M runtime — Tasks 2A/2B retain float32 I/O which the integer-only inference engine cannot run natively.
- The combination of P1 pruning + full INT8 quantization achieved both the best accuracy and smallest size across all 11 configurations.

| Model | Size | Reduction | Test Accuracy |
|-------|------|-----------|---------------|
| Baseline (float32) | 73,960 B | — | 76.9% |
| Pruning only (P1, whole s=0.30) | 73,960 B | 0% | 80.1% |
| Quantization only (full INT8) | 22,400 B | 69.7% | 76.6% |
| **P1 + full INT8 (deployed)** | **22,400 B** | **69.7%** | **80.1%** |

### Deployment

Camera: OV7670 initialised in QQVGA grayscale at 5fps, producing 160×120 frames. Center-cropped to 64×64 at offset (48, 28). Pixel values shifted from uint8 [0,255] to int8 [−128,127] in place when filling the input tensor — matches the model's quantization parameters (scale=1/255, zero_point=−128) without any floating-point arithmetic on-device.

Tensor arena: 100KB statically allocated, 16-byte aligned. `MicroMutableOpResolver<8>` registers only the ops the deployed graph uses (Conv2D, MaxPool2D, Reshape, FullyConnected, Softmax, Quantize, Dequantize, Mul) rather than AllOpsResolver, to keep flash footprint down. Inference is gated on a serial `'c'` command so a failed `Camera.readFrame()` cannot leave the board unresponsive.

`visualize.py` handshakes on the board's `System Ready` message, sends the capture trigger, then reconstructs the 64×64 frame and per-class INT8 scores from the serial stream.

RGB pipeline (`camera-classification/rgb/`) was explored but streaming 12,288 bytes/frame as ASCII decimal caused serial buffer overflows and channel misalignment. Switching to raw binary transmission fixed the communication layer, but the grayscale pipeline gave more reliable end-to-end results overall.

---

## Stack

| | |
|-|-|
| **Audio** | Edge Impulse, TFLite Micro, Arduino Nano 33 BLE Sense |
| **Camera** | TensorFlow 2.20, TFMOT, TFLite Micro, OV7670, Arduino IDE, pyserial |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Repo structure

| Path | Description |
|------|-------------|
| `audio-classification/classification.py` | 1D CNN training script (MFCC, Edge Impulse) |
| `audio-classification/nano_ble33_sense_microphone.ino` | Arduino inference sketch |
| `camera-classification/grayscale/compression.ipynb` | Pruning and quantization sweep |
| `camera-classification/grayscale/model_pruned_quant.tflite` | Deployed TFLite model (P1 + INT8) |
| `camera-classification/grayscale/model_prune_quant.h` | C header for Arduino |
| `camera-classification/grayscale/grayscale.ino` | Arduino inference + camera sketch |
| `camera-classification/grayscale/visualize.py` | Host-side capture and prediction visualiser |
| `camera-classification/grayscale/validate.py` | Cross-checks Arduino vs Python TFLite logits |
| `camera-classification/rgb/` | RGB experiment (serial comm issues documented in notes) |
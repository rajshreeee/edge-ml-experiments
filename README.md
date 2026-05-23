# Edge ML: Audio & Camera Classification

Demo video (audio + camera): [View online](https://abofi-my.sharepoint.com/:v:/g/personal/rajshree_rai_abo_fi/IQD0ob9OheecR6FP499iJQbXAa_xzoMxzSbm8qJ4twCOBgg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=91CimD) or download from [video.mp4](video.mp4)

## Setup

```bash
pip install -r requirements.txt
```

---

## Audio Classification

| File | Purpose |
|------|---------|
| [classification.py](audio-classification/classification.py) | Train the classification model |
| [nano_ble33_sense_microphone.ino](audio-classification/nano_ble33_sense_microphone.ino) | Arduino deployment |

---

## Camera Classification

### Grayscale (successful experiment)

| File | Purpose |
|------|---------|
| [compression.ipynb](camera-classification/grayscale/compression.ipynb) | Produces pruned + quantized model |
| [model_pruned_quant.tflite](camera-classification/grayscale/model_pruned_quant.tflite) | Exported TFLite model |
| [model_prune_quant.h](camera-classification/grayscale/model_prune_quant.h) | Header file for Arduino |
| [grayscale.ino](camera-classification/grayscale/grayscale.ino) | Arduino deployment |
| [visualize.py](camera-classification/grayscale/visualize.py) | Visualize predictions on device |
| [validate.py](camera-classification/grayscale/validate.py) | Validate model responses |

### RGB (experimental — serial communication issues)

| File | Purpose |
|------|---------|
| [rgb.py](camera-classification/rgb/rgb.py) | Model deployment |
| [sketch.ino](camera-classification/rgb/sketch.ino) | Arduino deployment |

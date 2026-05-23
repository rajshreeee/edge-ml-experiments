#!/usr/bin/env python3

import argparse
import re
import sys
import numpy as np

LABELS = ["paper", "rock", "scissors"]
IMG_H, IMG_W = 64, 64


def parse_serial_dump(text):
    """Pull the image bytes, Arduino logits, and Arduino prediction out of
    the serial output text."""
    m = re.search(
        r"---\s*START DATA\s*---(.*?)---\s*END DATA\s*---", text, re.S
    )
    if not m:
        raise ValueError(
            "No '--- START DATA --- ... --- END DATA ---' block found."
        )
    raw = m.group(1).strip().replace("\n", ",")
    pixels = [int(x) for x in raw.split(",") if x.strip()]
    expected = IMG_H * IMG_W
    if len(pixels) != expected:
        raise ValueError(
            f"Expected {expected} pixel values, got {len(pixels)}."
        )
    img = np.array(pixels, dtype=np.uint8).reshape(IMG_H, IMG_W)

    tail = text[m.end():]
    ard_logits = {}
    for label in LABELS:
        pm = re.search(rf"^\s*{label}\s*:\s*(-?\d+)\s*$", tail, re.M)
        if pm:
            ard_logits[label] = int(pm.group(1))
    pred_m = re.search(r">>\s*Prediction:\s*(\w+)", tail)
    arduino_pred = pred_m.group(1) if pred_m else None

    return img, ard_logits, arduino_pred


def run_python_inference(model_path, img_uint8):
    import tensorflow as tf  

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    if in_d["dtype"] != np.int8:
        raise SystemExit(
            f"Expected int8 input, model has {in_d['dtype']}. "
        )

    x_int8 = (img_uint8.astype(np.int16) - 128).astype(np.int8)
    x_int8 = x_int8.reshape(1, IMG_H, IMG_W, 1)

    interp.set_tensor(in_d["index"], x_int8)
    interp.invoke()
    py_out = interp.get_tensor(out_d["index"])[0]
    return py_out, in_d, out_d


def read_from_serial(timeout_s=20):
    import time
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial not installed")

    ser = serial.Serial(timeout=1)
    time.sleep(2) 
    ser.reset_input_buffer()
    ser.write(b"c")
    print("Sent 'c' — waiting for frame...", file=sys.stderr)

    buf, seen_start, seen_end, seen_pred = [], False, False, False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore")
        if not line:
            continue
        buf.append(line)
        if "START DATA" in line: seen_start = True
        if "END DATA"   in line: seen_end = True
        if seen_end and ">> Prediction:" in line:
            seen_pred = True
            break
    ser.close()
    if not (seen_start and seen_end and seen_pred):
        raise RuntimeError("Incomplete frame captured from serial.")
    return "".join(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=False, help="Path to .tflite model", default="./model_pruned_quant.tflite")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--serial", required=False, help="Path to a saved serial-log text file", default="./serial.txt")
    ap.add_argument("--save-image", help="Optional: save the captured frame as PNG", default="./captured_frame.png")
    args = ap.parse_args()

    text = open(args.serial).read()

    img, ard_logits, ard_pred = parse_serial_dump(text)
    print(f"Image parsed: shape={img.shape}, "
          f"min={img.min()}, max={img.max()}, mean={img.mean():.1f}")

    if args.save_image:
        try:
            from PIL import Image
            Image.fromarray(img, mode="L").save(args.save_image)
            print(f"Saved frame to {args.save_image}")
        except ImportError:
            print("(install Pillow to use --save-image)", file=sys.stderr)

    py_out, in_d, out_d = run_python_inference(args.model, img)
    py_pred = LABELS[int(np.argmax(py_out))]

    print()
    print(f"Input  quant: scale={in_d['quantization'][0]:.6g}, "
          f"zero_point={in_d['quantization'][1]}")
    print(f"Output quant: scale={out_d['quantization'][0]:.6g}, "
          f"zero_point={out_d['quantization'][1]}")
    print()
    print(f"{'Label':<10}{'Arduino':>10}{'Python':>10}{'Δ':>6}")
    print("-" * 36)
    for i, lbl in enumerate(LABELS):
        a = ard_logits.get(lbl)
        p = int(py_out[i])
        d = (p - a) if a is not None else None
        a_str = str(a) if a is not None else "n/a"
        d_str = f"{d:+d}" if d is not None else "?"
        print(f"{lbl:<10}{a_str:>10}{p:>10}{d_str:>6}")

    print()
    print(f"Arduino prediction: {ard_pred}")
    print(f"Python  prediction: {py_pred}")
    if ard_pred is None:
        print("Couldn't parse Arduino prediction line.")
    elif ard_pred == py_pred:
        print("Predictions MATCH")
    else:
        print("Predictions DIFFER")


if __name__ == "__main__":
    main()
import serial
import time
import numpy as np
import matplotlib.pyplot as plt


class CameraVisualizer:
    def __init__(self, port='/dev/cu.usbmodem101', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            print(" Waiting for Arduino to boot after USB reset (~5s)...")
            time.sleep(5)
            self.ser.reset_input_buffer()

            print(" Waiting for 'System Ready' handshake...")
            start = time.time()
            while time.time() - start < 10:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"  Arduino: {line}")
                    if "System Ready" in line:
                        print(" Synced! Board is ready.")
                        return True
                time.sleep(0.05)

            print(" 'System Ready' not seen — flushing and proceeding...")
            self.ser.reset_input_buffer()
            return True

        except Exception as e:
            print(f" Connection failed: {e}")
            return False

    def capture_and_display(self):
        if not self.ser or not self.ser.is_open:
            print(" Not connected")
            return

        print("\n3 seconds until capture...")
        for i in range(3, 0, -1):
            print(f" {i}...")
            time.sleep(1)
        print("Sending capture command...\n")

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.write(b'c')
        self.ser.flush()

        print(" Receiving data from Arduino...")

        start_time = time.time()
        header_buf = b""
        got_header = False
        while time.time() - start_time < 10:
            if self.ser.in_waiting:
                header_buf += self.ser.read(self.ser.in_waiting)
            if b"--- START DATA ---" in header_buf:
                got_header = True
                break
            time.sleep(0.02)

        if not got_header:
            print(" Never received START DATA header.")
            print(header_buf.decode('utf-8', errors='ignore')[:300] or "[EMPTY]")
            return

        header_end = header_buf.find(b"--- START DATA ---") + len(b"--- START DATA ---")
        leftover = header_buf[header_end:].lstrip(b'\r\n')

        pixel_bytes = leftover
        start_time = time.time()
        while len(pixel_bytes) < 4096 and time.time() - start_time < 10:
            needed = 4096 - len(pixel_bytes)
            chunk = self.ser.read(needed)  
            pixel_bytes += chunk

        if len(pixel_bytes) < 4096:
            print(f" Only got {len(pixel_bytes)}/4096 pixel bytes.")
            return

        pixels = np.frombuffer(pixel_bytes[:4096], dtype=np.uint8)
        print(f" Got {len(pixels)} pixel bytes")

        raw_bytes = b""
        start_time = time.time()
        while time.time() - start_time < 10:
            if self.ser.in_waiting:
                raw_bytes += self.ser.read(self.ser.in_waiting)
            if b">> Prediction:" in raw_bytes:
                time.sleep(0.1)
                raw_bytes += self.ser.read(self.ser.in_waiting)
                break
            time.sleep(0.05)

        raw_text = raw_bytes.decode('utf-8', errors='ignore')

        scores = {}
        for line in raw_text.split('\n'):
            if ':' in line and 'Prediction' not in line and 'END' not in line and 'Prob' not in line:
                key, val = line.split(':', 1)
                try:
                    scores[key.strip()] = int(val.strip())
                except ValueError:
                    pass

        prediction = ""
        for line in raw_text.split('\n'):
            if ">> Prediction:" in line:
                prediction = line.split(">> Prediction:")[1].strip()
                break

        if not prediction:
            print(" Could not parse prediction.")
            print(raw_text[:300])
            return

        img = pixels.reshape(64, 64)
        self._show_results(img, prediction, scores)

    def _show_results(self, img, prediction, scores):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

        ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'Prediction: {prediction.upper()}', fontweight='bold', fontsize=14)
        ax1.axis('off')

        labels = list(scores.keys())
        values = [scores.get(l, 0) for l in labels]
        colors = ['#2ecc71' if l == prediction else '#95a5a6' for l in labels]

        ax2.bar(labels, values, color=colors, alpha=0.8, width=0.5)
        ax2.set_ylim(-130, 130)
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_ylabel('Quantized Score (int8)')
        ax2.set_title('Model Output Scores')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(" Serial closed")


if __name__ == "__main__":
    viz = CameraVisualizer()
    if viz.connect():
        viz.capture_and_display()
        viz.close()
import serial
import numpy as np
from PIL import Image
import time

PORT = '/dev/tty.usbmodem101' 
EXPECTED_BYTES = 64 * 64 * 3 

def capture():
    try:
        ser = serial.Serial(PORT, 115200, timeout=5)
        time.sleep(2)
        ser.reset_input_buffer()
        
        print("Triggering...")
        ser.write(b'c')

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "READY" in line:
                break
        
        print("Reading binary data...")
        raw_data = ser.read(EXPECTED_BYTES)
        
        if len(raw_data) == EXPECTED_BYTES:
            img_np = np.frombuffer(raw_data, dtype=np.uint8).reshape((64, 64, 3))
            img = Image.fromarray(img_np)
            img.save("output.png")
            print("Done! Check output.png")
            
            result = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"Model says: {result}")
        else:
            print(f"Failed! Only got {len(raw_data)} bytes.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()

if __name__ == "__main__":
    capture()
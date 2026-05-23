#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_OV767X.h>
#include "model_rgb.h"

extern unsigned char model_tflite[] alignas(16);
extern unsigned int model_tflite_len;

namespace {
  const tflite::Model* tflModel = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 100 * 1024; 
  uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
}

const char* LABELS[] = {"paper", "rock", "scissors"};

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!Camera.begin(QQVGA, RGB565, 5)) {
    Serial.println("Camera Init Failed");
    while (1);
  }

  tflModel = tflite::GetModel(model_tflite);
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();        
  resolver.AddFullyConnected(); 
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  static tflite::MicroInterpreter static_interpreter(
      tflModel, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Allocation Failed!");
    while (1);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System Ready! Send 'c' to capture.");
}

void captureAndClassify() {
  static uint16_t raw_frame[160 * 120]; 
  Camera.readFrame((uint8_t*)raw_frame);

  int startX = 48; 
  int startY = 28; 

  Serial.println("READY");
  delay(10); 

  for (int y = 0; y < 64; y++) {
    for (int x = 0; x < 64; x++) {
      int pixel_idx = ((startY + y) * 160) + (startX + x);
      uint16_t pixel = raw_frame[pixel_idx];
      pixel = (pixel >> 8) | (pixel << 8); 

      uint8_t r = (pixel >> 11) << 3;
      uint8_t g = (pixel >> 5 & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;

      Serial.write(r);
      Serial.write(g);
      Serial.write(b);

      int tensor_idx = (y * 64 + x) * 3;
      input->data.int8[tensor_idx + 0] = (int8_t)(r - 128);
      input->data.int8[tensor_idx + 1] = (int8_t)(g - 128);
      input->data.int8[tensor_idx + 2] = (int8_t)(b - 128);
    }
  }
  
  if (interpreter->Invoke() == kTfLiteOk) {
    int8_t max_val = -128;
    int winner = 0;
    for (int i = 0; i < 3; i++) {
      if (output->data.int8[i] > max_val) {
        max_val = output->data.int8[i];
        winner = i;
      }
    }
    Serial.print("RESULT:");
    Serial.println(LABELS[winner]);
  }
}

void loop() {
  if (Serial.available() > 0) {
    if (Serial.read() == 'c') captureAndClassify();
  }
}
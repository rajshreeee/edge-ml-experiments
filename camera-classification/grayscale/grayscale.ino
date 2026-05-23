#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_OV767X.h>

extern const unsigned char model[] alignas(16);
#include "model_prune_quant.h"

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

  if (!Camera.begin(QQVGA, GRAYSCALE, 5)) {
    Serial.println("Camera Init Failed!");
    while (1);
  }

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model Schema Mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();        
  resolver.AddFullyConnected(); 
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMul();

  static tflite::MicroInterpreter static_interpreter(
      tflModel, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor Allocation Failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System Ready. Send 'c' to Capture, Classify, and View Image.");
}

void captureAndClassify() {
  static uint8_t raw_frame[160 * 120]; 
  Camera.readFrame(raw_frame);

  int startX = 48; // (160 - 64) / 2
  int startY = 28; // (120 - 64) / 2

  Serial.println("--- START DATA ---");
  for (int y = 0; y < 64; y++) {
    for (int x = 0; x < 64; x++) {
      int pixel_idx = ((startY + y) * 160) + (startX + x);
      uint8_t pixel_val = raw_frame[pixel_idx];
      
      Serial.print(pixel_val);
      if ((y * 64 + x) < 4095) Serial.print(",");

      input->data.int8[y * 64 + x] = (int8_t)(pixel_val - 128);
    }
  }
  Serial.println("\n--- END DATA ---");

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference Failed!");
    return;
  }

  int8_t max_score = -128;
  int winner = 0;
  
  Serial.println("--- Probabilities ---");
  for (int i = 0; i < 3; i++) {
    int8_t score = output->data.int8[i];
    Serial.print(LABELS[i]); Serial.print(": "); Serial.println(score);
    if (score > max_score) {
      max_score = score;
      winner = i;
    }
  }

  Serial.print(">> Prediction: ");
  Serial.println(LABELS[winner]);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'c') {
      captureAndClassify();
    }
  }
}
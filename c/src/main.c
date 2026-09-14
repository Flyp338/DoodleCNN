#include "../include/layers.h"
#include "../include/model_loader.h"
#include "../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>


#define NUM_CLASSES 10
#define IMG_SIZE 28

// Helper to find index of maximum value
int argmax(const Tensor *logits) {
  int max_idx = 0;
  float max_val = logits->data[0];
  for (size_t i = 1; i < logits->total_size; i++) {
    if (logits->data[i] > max_val) {
      max_val = logits->data[i];
      max_idx = (int)i;
    }
  }
  return max_idx;
}

// Forward Pass: (Input [1, 1, 28, 28]) -> Output Class Logits [1, 10]
Tensor *forward(const Tensor *input, const ModelWeights *weights) {
  // 1. Conv1 + ReLU + Pool1
  Tensor *c1 = conv2d(input, weights->conv1_weight, weights->conv1_bias, 1, 1);
  relu(c1);
  Tensor *p1 = maxpool2d(c1, 2, 2);
  free_tensor(c1);

  // 2. Conv2 + ReLU + Pool2
  Tensor *c2 = conv2d(p1, weights->conv2_weight, weights->conv2_bias, 1, 1);
  free_tensor(p1);
  relu(c2);
  Tensor *p2 = maxpool2d(c2, 2, 2);
  free_tensor(c2);

  // 3. Flatten -> FC1 + ReLU
  Tensor *flat = flatten(p2);
  free_tensor(p2);
  Tensor *fc1_out = linear(flat, weights->fc1_weight, weights->fc1_bias);
  free_tensor(flat);
  relu(fc1_out);

  // 4. FC2 (Logits)
  Tensor *logits = linear(fc1_out, weights->fc2_weight, weights->fc2_bias);
  free_tensor(fc1_out);

  return logits;
}

int main() {
  printf("Initializing C Inference Engine...\n");

  // Load exported weights
  ModelWeights *weights = load_model_weights("models/exported/model.bin");
  if (!weights) {
    fprintf(stderr, "Failed to load model binary.\n");
    return 1;
  }
  printf("Model weights successfully loaded.\n");

  // Create a dummy zero/input test tensor: [1, 1, 28, 28]
  size_t in_shape[4] = {1, 1, IMG_SIZE, IMG_SIZE};
  Tensor *input = create_tensor(4, in_shape);

  // Run forward pass
  printf("Executing forward pass...\n");
  Tensor *logits = forward(input, weights);

  // Apply Softmax for probabilities
  softmax(logits);

  int predicted_class = argmax(logits);
  printf("\nInference Complete!\n");
  printf("Predicted Class ID: %d (Confidence: %.2f%%)\n", predicted_class,
         logits->data[predicted_class] * 100.0f);

  // Clean up memory
  free_tensor(input);
  free_tensor(logits);
  free_model_weights(weights);

  return 0;
}
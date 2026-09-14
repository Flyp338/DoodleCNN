#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/model_loader.h"
#include "../include/layers.h"

#define NUM_CLASSES 10
#define IMG_SIZE 28

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

Tensor *forward(const Tensor *input, const ModelWeights *weights) {
    Tensor *c1 = conv2d(input, weights->conv1_weight, weights->conv1_bias, 1, 1);
    relu(c1);
    Tensor *p1 = maxpool2d(c1, 2, 2);
    free_tensor(c1);

    Tensor *c2 = conv2d(p1, weights->conv2_weight, weights->conv2_bias, 1, 1);
    free_tensor(p1);
    relu(c2);
    Tensor *p2 = maxpool2d(c2, 2, 2);
    free_tensor(c2);

    Tensor *flat = flatten(p2);
    free_tensor(p2);
    Tensor *fc1_out = linear(flat, weights->fc1_weight, weights->fc1_bias);
    free_tensor(flat);
    relu(fc1_out);

    Tensor *logits = linear(fc1_out, weights->fc2_weight, weights->fc2_bias);
    free_tensor(fc1_out);

    return logits;
}

int main() {
    printf("--- Running C Model Verification ---\n");

    ModelWeights *weights = load_model_weights("models/exported/model.bin");
    if (!weights) {
        fprintf(stderr, "Error loading model weights!\n");
        return 1;
    }

    // Allocate input tensor for 1 image
    size_t shape[4] = {1, 1, IMG_SIZE, IMG_SIZE};
    Tensor *input = create_tensor(4, shape);

    // Run forward pass
    Tensor *logits = forward(input, weights);
    softmax(logits);

    printf("Class Probabilities:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("  Class %d: %6.2f%%\n", i, logits->data[i] * 100.0f);
    }

    printf("\nTop Prediction: Class %d\n", argmax(logits));

    free_tensor(input);
    free_tensor(logits);
    free_model_weights(weights);
    return 0;
}
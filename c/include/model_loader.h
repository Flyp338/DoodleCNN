#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "tensor.h"
#include <stdint.h>

typedef struct {
    Tensor *conv1_weight; // [16, 1, 3, 3]
    Tensor *conv1_bias;   // [16]
    Tensor *conv2_weight; // [32, 16, 3, 3]
    Tensor *conv2_bias;   // [32]
    Tensor *fc1_weight;   // [128, 1568]
    Tensor *fc1_bias;     // [128]
    Tensor *fc2_weight;   // [10, 128]
    Tensor *fc2_bias;     // [10]
} ModelWeights;

ModelWeights *load_model_weights(const char *filepath);
void free_model_weights(ModelWeights *weights);

#endif // MODEL_LOADER_H
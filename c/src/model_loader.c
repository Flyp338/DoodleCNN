#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/model_loader.h"

static Tensor *read_tensor_from_file(FILE *f) {
    uint32_t ndim = 0;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1) return NULL;

    size_t *shape = (size_t *)malloc(ndim * sizeof(size_t));
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim_val = 0;
        if (fread(&dim_val, sizeof(uint32_t), 1, f) != 1) {
            free(shape);
            return NULL;
        }
        shape[i] = (size_t)dim_val;
    }

    Tensor *tensor = create_tensor((size_t)ndim, shape);
    free(shape);

    if (!tensor) return NULL;

    size_t read_count = fread(tensor->data, sizeof(float), tensor->total_size, f);
    if (read_count != tensor->total_size) {
        free_tensor(tensor);
        return NULL;
    }

    return tensor;
}

ModelWeights *load_model_weights(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open binary weight file at %s\n", filepath);
        return NULL;
    }

    ModelWeights *weights = (ModelWeights *)malloc(sizeof(ModelWeights));
    if (!weights) {
        fclose(f);
        return NULL;
    }

    weights->conv1_weight = read_tensor_from_file(f);
    weights->conv1_bias   = read_tensor_from_file(f);
    weights->conv2_weight = read_tensor_from_file(f);
    weights->conv2_bias   = read_tensor_from_file(f);
    weights->fc1_weight   = read_tensor_from_file(f);
    weights->fc1_bias     = read_tensor_from_file(f);
    weights->fc2_weight   = read_tensor_from_file(f);
    weights->fc2_bias     = read_tensor_from_file(f);

    fclose(f);

    if (!weights->conv1_weight || !weights->conv1_bias ||
        !weights->conv2_weight || !weights->conv2_bias ||
        !weights->fc1_weight   || !weights->fc1_bias   ||
        !weights->fc2_weight   || !weights->fc2_bias) {
        fprintf(stderr, "Error: Failed to read complete binary payload.\n");
        free_model_weights(weights);
        return NULL;
    }

    return weights;
}

void free_model_weights(ModelWeights *weights) {
    if (weights) {
        free_tensor(weights->conv1_weight);
        free_tensor(weights->conv1_bias);
        free_tensor(weights->conv2_weight);
        free_tensor(weights->conv2_bias);
        free_tensor(weights->fc1_weight);
        free_tensor(weights->fc1_bias);
        free_tensor(weights->fc2_weight);
        free_tensor(weights->fc2_bias);
        free(weights);
    }
}
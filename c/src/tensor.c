#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"

Tensor *create_tensor(size_t ndim, const size_t *shape){
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if (!tensor){
        fprintf(stderr, "Error: Failed to allocate memory for Tensor struct.\n");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        fprintf(stderr, "Error: Failed to allocate memory for Tensor shape.\n");
        free(tensor);
        return NULL;
    }
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++){
        tensor->shape[i] = shape[i];
        total *= shape[i];
    }
    tensor->total_size = total;
    tensor->data = (float *)calloc(total, sizeof(float));
    if (!tensor->data) {
        fprintf(stderr, "Error: Failed to allocate memory for Tensor data.\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    return tensor;
}

void free_tensor(Tensor *tensor){
    if (tensor){
        if (tensor->data) free(tensor->data);
        if (tensor->shape) free(tensor->shape);
        free(tensor);
    }
}

void print_tensor_shape(const Tensor *tensor) {
    if (!tensor) return;
    printf("Tensor Shape: [");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu%s", tensor->shape[i], (i == tensor->ndim - 1) ? "" : ", ");
    }
    printf("] (Total elements: %zu)\n", tensor->total_size);
}
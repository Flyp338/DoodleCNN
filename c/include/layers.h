#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"

// Layer prototypes
Tensor *conv2d(const Tensor *input, const Tensor *weights, const Tensor *bias, size_t padding, size_t stride);
Tensor *maxpool2d(const Tensor *input, size_t kernel_size, size_t stride);
Tensor *flatten(const Tensor *input);
Tensor *linear(const Tensor *input, const Tensor *weights, const Tensor *bias);
void relu(Tensor *tensor);
void softmax(Tensor *tensor);

#endif // LAYERS_H
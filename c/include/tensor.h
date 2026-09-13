#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct{
    float *data;
    size_t *shape;
    size_t ndim;
    size_t total_size;
} Tensor;


Tensor *create_tensor(size_t ndim, const size_t *shape);
void free_tensor(Tensor *tensor);
void print_tensor_shape(const Tensor *tensor);

#endif //Tensor
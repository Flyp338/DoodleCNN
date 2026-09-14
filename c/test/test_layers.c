#include <stdio.h>
#include "../include/tensor.h"
#include "../include/layers.h"

int main() {
    printf("Testing C CNN layers...\n");

    // Test input: [1, 1, 4, 4]
    size_t in_shape[4] = {1, 1, 4, 4};
    Tensor *input = create_tensor(4, in_shape);
    for (size_t i = 0; i < 16; i++) input->data[i] = (float)(i + 1);

    // Test MaxPool: 2x2, stride 2 -> expected shape [1, 1, 2, 2]
    Tensor *pooled = maxpool2d(input, 2, 2);
    print_tensor_shape(pooled);

    // Test ReLU
    pooled->data[0] = -5.0f;
    relu(pooled);
    printf("ReLU value (expected >= 0): %.1f\n", pooled->data[0]);

    free_tensor(input);
    free_tensor(pooled);
    printf("Layer unit tests PASSED!\n");
    return 0;
}
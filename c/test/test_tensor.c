#include "../include/tensor.h"
#include <stdio.h>


int main() {
  printf("Testing Tensor creation...\n");
  size_t shape[4] = {1, 16, 28, 28};
  Tensor *t = create_tensor(4, shape);

  if (t != NULL && t->total_size == 12544) {
    print_tensor_shape(t);
    printf("Tensor creation test PASSED!\n");
  } else {
    printf("Tensor creation test FAILED!\n");
  }

  free_tensor(t);
  return 0;
}
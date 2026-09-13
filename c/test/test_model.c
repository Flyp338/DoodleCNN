#include <stdio.h>
#include "../include/model_loader.h"

int main() {
    printf("Loading exported model binary...\n");
    ModelWeights *weights = load_model_weights("models/exported/model.bin");

    if (weights != NULL) {
        printf("Model weights loaded successfully!\n");
        printf("conv1_weight shape: "); print_tensor_shape(weights->conv1_weight);
        printf("fc2_weight shape:   "); print_tensor_shape(weights->fc2_weight);
        free_model_weights(weights);
    } else {
        printf("FAILED to load model weights!\n");
    }

    return 0;
}
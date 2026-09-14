#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/layers.h"

// 1. Conv2D Implementation (Input shape: [1, C_in, H, W], Weights shape: [C_out, C_in, K_h, K_w])
Tensor *conv2d(const Tensor *input, const Tensor *weights, const Tensor *bias, size_t padding, size_t stride) {
    size_t in_channels = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t out_channels = weights->shape[0];
    size_t k_h = weights->shape[2];
    size_t k_w = weights->shape[3];

    size_t out_h = (in_h + 2 * padding - k_h) / stride + 1;
    size_t out_w = (in_w + 2 * padding - k_w) / stride + 1;

    size_t out_shape[4] = {1, out_channels, out_h, out_w};
    Tensor *output = create_tensor(4, out_shape);

    for (size_t oc = 0; oc < out_channels; oc++) {
        float b = bias ? bias->data[oc] : 0.0f;
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
                float sum = b;
                for (size_t ic = 0; ic < in_channels; ic++) {
                    for (size_t kh = 0; kh < k_h; kh++) {
                        for (size_t kw = 0; kw < k_w; kw++) {
                            int ih = (int)(oh * stride + kh) - (int)padding;
                            int iw = (int)(ow * stride + kw) - (int)padding;

                            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                                size_t in_idx = ic * (in_h * in_w) + ih * in_w + iw;
                                size_t w_idx = oc * (in_channels * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;
                                sum += input->data[in_idx] * weights->data[w_idx];
                            }
                        }
                    }
                }
                size_t out_idx = oc * (out_h * out_w) + oh * out_w + ow;
                output->data[out_idx] = sum;
            }
        }
    }
    return output;
}

// 2. MaxPool2D Implementation
Tensor *maxpool2d(const Tensor *input, size_t kernel_size, size_t stride) {
    size_t channels = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t out_h = (in_h - kernel_size) / stride + 1;
    size_t out_w = (in_w - kernel_size) / stride + 1;

    size_t out_shape[4] = {1, channels, out_h, out_w};
    Tensor *output = create_tensor(4, out_shape);

    for (size_t c = 0; c < channels; c++) {
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
                float max_val = -1e9f;
                for (size_t kh = 0; kh < kernel_size; kh++) {
                    for (size_t kw = 0; kw < kernel_size; kw++) {
                        size_t ih = oh * stride + kh;
                        size_t iw = ow * stride + kw;
                        size_t in_idx = c * (in_h * in_w) + ih * in_w + iw;
                        if (input->data[in_idx] > max_val) {
                            max_val = input->data[in_idx];
                        }
                    }
                }
                size_t out_idx = c * (out_h * out_w) + oh * out_w + ow;
                output->data[out_idx] = max_val;
            }
        }
    }
    return output;
}

// 3. Flatten (Reshapes 4D [1, C, H, W] tensor to 2D [1, C*H*W])
Tensor *flatten(const Tensor *input) {
    size_t out_shape[2] = {1, input->total_size};
    Tensor *output = create_tensor(2, out_shape);
    for (size_t i = 0; i < input->total_size; i++) {
        output->data[i] = input->data[i];
    }
    return output;
}

// 4. Fully Connected (Linear) Layer (Weights shape: [out_features, in_features])
Tensor *linear(const Tensor *input, const Tensor *weights, const Tensor *bias) {
    size_t out_features = weights->shape[0];
    size_t in_features = weights->shape[1];

    size_t out_shape[2] = {1, out_features};
    Tensor *output = create_tensor(2, out_shape);

    for (size_t o = 0; o < out_features; o++) {
        float sum = bias ? bias->data[o] : 0.0f;
        for (size_t i = 0; i < in_features; i++) {
            sum += input->data[i] * weights->data[o * in_features + i];
        }
        output->data[o] = sum;
    }
    return output;
}

// 5. In-place ReLU Activation
void relu(Tensor *tensor) {
    for (size_t i = 0; i < tensor->total_size; i++) {
        if (tensor->data[i] < 0.0f) {
            tensor->data[i] = 0.0f;
        }
    }
}

// 6. In-place Softmax (Converts logits to normalized probabilities)
void softmax(Tensor *tensor) {
    float max_val = tensor->data[0];
    for (size_t i = 1; i < tensor->total_size; i++) {
        if (tensor->data[i] > max_val) max_val = tensor->data[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] = expf(tensor->data[i] - max_val);
        sum += tensor->data[i];
    }

    for (size_t i = 0; i < tensor->total_size; i++) {
        tensor->data[i] /= sum;
    }
}
#pragma once
#include <torch/extension.h>
#include <vector>

namespace detr {
at::Tensor ga_forward_cuda(
    const at::Tensor& query,
    const at::Tensor& key);

std::tuple<at::Tensor, at::Tensor> ga_backward_cuda(
    const at::Tensor& dw,
    const at::Tensor& query,
    const at::Tensor& key);

at::Tensor ga_map_forward_cuda(
    const at::Tensor& weight,
    const at::Tensor& value);

std::tuple<at::Tensor, at::Tensor> ga_map_backward_cuda(
    const at::Tensor& dout,
    const at::Tensor& weight,
    const at::Tensor& value);


at::Tensor ga_forward(const at::Tensor& query,
                      const at::Tensor& key) {
    if (query.device().is_cuda() && key.device().is_cuda()) {
  #ifdef WITH_CUDA
      return ga_forward_cuda(query, key);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
      AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> ga_backward(const at::Tensor& dw,
                                               const at::Tensor& query,
                                               const at::Tensor& key) {
    if (dw.device().is_cuda()) {
  #ifdef WITH_CUDA
      return ga_backward_cuda(dw, query, key);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor ga_map_forward(const at::Tensor& weight,
                          const at::Tensor& value) {
    if (weight.device().is_cuda() && value.device().is_cuda()) {
  #ifdef WITH_CUDA
      return ga_map_forward_cuda(weight, value);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> ga_map_backward(const at::Tensor& dout,
                                                   const at::Tensor& weight,
                                                   const at::Tensor& value) {
    if (dout.device().is_cuda() && value.device().is_cuda()) {
  #ifdef WITH_CUDA
      return ga_map_backward_cuda(dout, weight, value);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

}  // namespace detr
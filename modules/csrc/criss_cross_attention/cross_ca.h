#pragma once
#include <torch/extension.h>
#include <vector>

namespace detr {

at::Tensor cross_ca_weight_forward_cuda(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& point);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cross_ca_weight_backward_cuda(
    const at::Tensor& dw,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& point);

at::Tensor cross_ca_map_forward_cuda(
    const at::Tensor& weight,
    const at::Tensor& value,
    const at::Tensor& point);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cross_ca_map_backward_cuda(
    const at::Tensor& dout,
    const at::Tensor& weight,
    const at::Tensor& value,
    const at::Tensor& point);


at::Tensor cross_ca_weight_forward(const at::Tensor& query,
                                   const at::Tensor& key,
                                   const at::Tensor& point) {
    if (query.device().is_cuda()) {
  #ifdef WITH_CUDA
      return cross_ca_weight_forward_cuda(query, key, point);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cross_ca_weight_backward(const at::Tensor& dw,
                                                                        const at::Tensor& query,
                                                                        const at::Tensor& key,
                                                                        const at::Tensor& point) {
    if (dw.device().is_cuda()) {
  #ifdef WITH_CUDA
      return cross_ca_weight_backward_cuda(dw, query, key, point);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

at::Tensor cross_ca_map_forward(const at::Tensor& weight,
                                const at::Tensor& value,
                                const at::Tensor& point) {
    if (weight.device().is_cuda()) {
  #ifdef WITH_CUDA
      return cross_ca_map_forward_cuda(weight, value, point);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cross_ca_map_backward(const at::Tensor& dout,
                                                                     const at::Tensor& weight,
                                                                     const at::Tensor& value,
                                                                     const at::Tensor& point) {
    if (dout.device().is_cuda()) {
  #ifdef WITH_CUDA
      return cross_ca_map_backward_cuda(dout, weight, value, point);
  #else
      AT_ERROR("Not compiled with GPU support");
  #endif
  }
    AT_ERROR("Not implemented on the CPU");
}

}  // namespace detr
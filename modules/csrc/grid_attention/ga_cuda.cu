#include "common_cuda_helper.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

template <typename T>
__global__ void ga_forward_kernel(
    const int nthreads, const T* query, const T* key, T* weight,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_q, const int w_q,
    const int h_k, const int w_k) {
  int sp_q = h_q * w_q;
  int sp_k = h_k * w_k;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_k) / static_cast<T>(w_q);
  T step_h = static_cast<T>(h_k) / static_cast<T>(h_q);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_q;
    int gh = (index / w_q) % h_q;
    int c = (index / w_q / h_q) % len;
    int n = index / w_q / h_q / len;

    int bin_x1 = ceil(static_cast<T>(gw) * step_w);
    int bin_y1 = ceil(static_cast<T>(gh) * step_h);

    int bin_x_idx = c % local_attn_w;
    int bin_y_idx = (c / local_attn_w) % local_attn_h;

    int bin_x = bin_x1 + bin_x_idx;
    int bin_y = bin_y1 + bin_y_idx;

    bin_x = min(max(bin_x, 0), w_k);
    bin_y = min(max(bin_y, 0), h_k);

    for (int plane = 0; plane < channels; ++plane) {
      T _q = query[(n * channels + plane) * sp_q + gh * w_q + gw];
      T _k = key[(n * channels + plane) * sp_k + bin_y * w_k + bin_x];
      weight[(n * len + c) * sp_q + gh * w_q + gw] += _q * _k;
    }
  }
}

template <typename T>
__global__ void ga_backward_kernel_q(
    const int nthreads, const T* dw, const T* query, const T* key, T* dq,
    const int local_attn_h, const int local_attn_w, const int num, 
    const int channels, const int h_q, const int w_q,
    const int h_k, const int w_k) {
  int sp_q = h_q * w_q;
  int sp_k = h_k * w_k;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_k) / static_cast<T>(w_q);
  T step_h = static_cast<T>(h_k) / static_cast<T>(h_q);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_q;
    int gh = (index / w_q) % h_q;
    int c = (index / w_q / h_q) % channels;
    int n = index / w_q / h_q / channels;

    int bin_x1 = ceil(static_cast<T>(gw) * step_w);
    int bin_y1 = ceil(static_cast<T>(gh) * step_h);
    int bin_x2 = bin_x1 + local_attn_w;
    int bin_y2 = bin_y1 + local_attn_h;

    bin_x1 = min(max(bin_x1, 0), w_k);
    bin_y1 = min(max(bin_y1, 0), h_k);
    bin_x2 = min(max(bin_x2, 0), w_k);
    bin_y2 = min(max(bin_y2, 0), h_k);

    for (int h = bin_y1; h < bin_y2; ++h) {
      for (int w = bin_x1; w < bin_x2; ++w) {
        T _dw = dw[(n * len + (h - bin_y1) * local_attn_w + (w - bin_x1)) * sp_q + gh * w_q + gw];
        T _k = key[(n * channels + c) * sp_k + h * w_k + w];
        dq[(n * channels + c) * sp_q + gh * w_q + gw] += _dw * _k;
      }
    }
  }
}

template <typename T>
__global__ void ga_backward_kernel_k(
    const int nthreads, const T* dw, const T* query, const T* key, T* dk,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_q, const int w_q,
    const int h_k, const int w_k) {
  int sp_q = h_q * w_q;
  int sp_k = h_k * w_k;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_k) / static_cast<T>(w_q);
  T step_h = static_cast<T>(h_k) / static_cast<T>(h_q);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, kw, kh) is an element in the output
    int kw = index % w_k;
    int kh = (index / w_k) % h_k;
    int c = (index / w_k / h_k) % channels;
    int n = index / w_k / h_k / channels;

    int q_x = floor(static_cast<T>(kw) / step_w);
    int q_y = floor(static_cast<T>(kh) / step_h);
    int ref_x = ceil(static_cast<T>(q_x) * step_w);
    int ref_y = ceil(static_cast<T>(q_y) * step_h);
    if (!(((ref_x <= kw) && (kw < ref_x + local_attn_w)) && ((ref_y <= kh) && (kh < ref_y + local_attn_h)))) {
      continue;
    }

    int channel_idx = (kw - ref_x) + (kh - ref_y) * local_attn_w;

    T _dw = dw[(n * len + channel_idx) * sp_q + q_y * w_q + q_x];
    T _q = query[(n * channels + c) * sp_q + q_y * w_q + q_x];
    dk[(n * channels + c) * sp_k + kh * w_k + kw] += _dw * _q;
  }
}

template <typename T>
__global__ void ga_map_forward_kernel(
    const int nthreads, const T* weight, const T* value, T* out,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_w, const int w_w,
    const int h_v, const int w_v) {
  int sp_w = h_w * w_w;
  int sp_v = h_v * w_v;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_v) / static_cast<T>(w_w);
  T step_h = static_cast<T>(h_v) / static_cast<T>(h_w);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_w;
    int gh = (index / w_w) % h_w;
    int c = (index / w_w / h_w) % channels;
    int n = index / w_w / h_w / channels;

    int bin_x1 = ceil(static_cast<T>(gw) * step_w);
    int bin_y1 = ceil(static_cast<T>(gh) * step_h);
    int bin_x2 = bin_x1 + local_attn_w;
    int bin_y2 = bin_y1 + local_attn_h;

    bin_x1 = min(max(bin_x1, 0), w_v);
    bin_y1 = min(max(bin_y1, 0), h_v);
    bin_x2 = min(max(bin_x2, 0), w_v);
    bin_y2 = min(max(bin_y2, 0), h_v);

    for (int h = bin_y1; h < bin_y2; ++h) {
      for (int w = bin_x1; w < bin_x2; ++w) {
        T _w = weight[(n * len + (h - bin_y1) * local_attn_w + (w - bin_x1)) * sp_w + gh * w_w + gw];
        T _v = value[(n * channels + c) * sp_v + h * w_v + w];
        out[(n * channels + c) * sp_w + gh * w_w + gw] += _w * _v;
      }
    }
  }
}

template <typename T>
__global__ void ga_map_backward_kernel_w(
    const int nthreads, const T* dout, const T* weight, const T* value, T* dw,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_out, const int w_out,
    const int h_v, const int w_v) {
  int sp_out = h_out * w_out;
  int sp_v = h_v * w_v;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_v) / static_cast<T>(w_out);
  T step_h = static_cast<T>(h_v) / static_cast<T>(h_out);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_out;
    int gh = (index / w_out) % h_out;
    int c = (index / w_out / h_out) % len;
    int n = index / w_out / h_out / len;

    int bin_x1 = ceil(static_cast<T>(gw) * step_w);
    int bin_y1 = ceil(static_cast<T>(gh) * step_h);

    int bin_x_idx = c % local_attn_w;
    int bin_y_idx = (c / local_attn_w) % local_attn_h;

    int bin_x = bin_x1 + bin_x_idx;
    int bin_y = bin_y1 + bin_y_idx;

    bin_x = min(max(bin_x, 0), w_v);
    bin_y = min(max(bin_y, 0), h_v);

    for (int plane = 0; plane < channels; ++plane) {
      T _dout = dout[(n * channels + plane) * sp_out + gh * w_out + gw];
      T _v = value[(n * channels + plane) * sp_v + bin_y * w_v + bin_x];
      dw[(n * len + c) * sp_out + gh * w_out + gw] += _dout * _v;
    }
  }
}

template <typename T>
__global__ void ga_map_backward_kernel_v(
    const int nthreads, const T* dout, const T* weight, const T* value, T* dv,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_out, const int w_out,
    const int h_v, const int w_v) {
  int sp_out = h_out * w_out;
  int sp_v = h_v * w_v;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_v) / static_cast<T>(w_out);
  T step_h = static_cast<T>(h_v) / static_cast<T>(h_out);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, vh, vw) is an element in the output
    int vw = index % w_v;
    int vh = (index / w_v) % h_v;
    int c = (index / w_v / h_v) % channels;
    int n = index / w_v / h_v / channels;

    int w_x = floor(static_cast<T>(vw) / step_w);
    int w_y = floor(static_cast<T>(vh) / step_h);
    int ref_x = ceil(static_cast<T>(w_x) * step_w);
    int ref_y = ceil(static_cast<T>(w_y) * step_h);
    if (!((ref_x <= vw && vw < (ref_x + local_attn_w))
      && (ref_y <= vh && vh < (ref_y + local_attn_h)))) {
      continue;
    }

    int channel_idx = (vw - ref_x) + (vh - ref_y) * local_attn_w;

    T _dout = dout[(n * channels + c) * sp_out + w_y * w_out + w_x];
    T _w = weight[(n * len + channel_idx) * sp_out + w_y * w_out + w_x];
    dv[(n * channels + c) * sp_v + vh * w_v + vw] += _dout * _w;
  }
}


namespace detr {

at::Tensor ga_forward_cuda(const at::Tensor& query,
                           const at::Tensor& key) {
  AT_ASSERTM(query.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(key.device().is_cuda(), "input must be a CUDA tensor");

  auto n = query.size(0);
  auto c = query.size(1);
  auto h_q = query.size(2);
  auto w_q = query.size(3);

  auto h_k = key.size(2);
  auto w_k = key.size(3);
  int local_attn_h = floor(h_k / h_q);
  int local_attn_w = floor(w_k / w_q);

  at::Tensor weight = at::zeros({n, local_attn_h * local_attn_w, h_q, w_q}, query.options());
  int cal_size = weight.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "ga_forward", [&] {
      ga_forward_kernel<scalar_t><<<GET_BLOCKS(cal_size), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size,
        query.contiguous().data_ptr<scalar_t>(),
        key.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_q, w_q, h_k, w_k);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return weight;
}

std::tuple<at::Tensor, at::Tensor> ga_backward_cuda(const at::Tensor& dw,
                                                    const at::Tensor& query,
                                                    const at::Tensor& key) {
  AT_ASSERTM(dw.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(query.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(key.device().is_cuda(), "input must be a CUDA tensor");

  auto n = query.size(0);
  auto c = query.size(1);
  auto h_q = query.size(2);
  auto w_q = query.size(3);

  auto h_k = key.size(2);
  auto w_k = key.size(3);
  int local_attn_h = floor(h_k / h_q);
  int local_attn_w = floor(w_k / w_q);

  at::Tensor dq = at::zeros_like(query);
  at::Tensor dk = at::zeros_like(key);
  int cal_size_q = dq.numel();
  int cal_size_k = dk.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "ga_backward_kernel_q", [&] {
      ga_backward_kernel_q<scalar_t><<<GET_BLOCKS(cal_size_q), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_q,
        dw.contiguous().data_ptr<scalar_t>(),
        query.contiguous().data_ptr<scalar_t>(),
        key.contiguous().data_ptr<scalar_t>(),
        dq.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_q, w_q, h_k, w_k);
  });

  AT_DISPATCH_FLOATING_TYPES(key.scalar_type(), "ga_backward_kernel_k", [&] {
      ga_backward_kernel_k<scalar_t><<<GET_BLOCKS(cal_size_k), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_k,
        dw.contiguous().data_ptr<scalar_t>(),
        query.contiguous().data_ptr<scalar_t>(),
        key.contiguous().data_ptr<scalar_t>(),
        dk.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_q, w_q, h_k, w_k);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(dq, dk);
}

at::Tensor ga_map_forward_cuda(const at::Tensor& weight,
                               const at::Tensor& value) {
  AT_ASSERTM(weight.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(value.device().is_cuda(), "input must be a CUDA tensor");

  auto n = value.size(0);
  auto c = value.size(1);
  auto h_v = value.size(2);
  auto w_v = value.size(3);

  auto h_w = weight.size(2);
  auto w_w = weight.size(3);
  int local_attn_h = floor(h_v / h_w);
  int local_attn_w = floor(w_v / w_w);

  at::Tensor out = at::zeros({n, c, h_w, w_w}, weight.options());
  int cal_size_out = out.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ga_map_forward_kernel", [&] {
      ga_map_forward_kernel<scalar_t><<<GET_BLOCKS(cal_size_out), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_out,
        weight.contiguous().data_ptr<scalar_t>(),
        value.contiguous().data_ptr<scalar_t>(),
        out.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_w, w_w, h_v, w_v);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

std::tuple<at::Tensor, at::Tensor> ga_map_backward_cuda(const at::Tensor& dout,
                                                       const at::Tensor& weight,
                                                       const at::Tensor& value) {
  AT_ASSERTM(dout.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(weight.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(value.device().is_cuda(), "input must be a CUDA tensor");

  auto n = dout.size(0);
  auto c = dout.size(1);
  auto h_out = dout.size(2);
  auto w_out = dout.size(3);

  auto h_v = value.size(2);
  auto w_v = value.size(3);
  int local_attn_h = floor(h_v / h_out);
  int local_attn_w = floor(w_v / w_out);

  at::Tensor dw = at::zeros_like(weight);
  at::Tensor dv = at::zeros_like(value);
  int cal_size_w = dw.numel();
  int cal_size_v = dv.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "ga_map_backward_kernel_w", [&] {
      ga_map_backward_kernel_w<scalar_t><<<GET_BLOCKS(cal_size_w), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_w,
        dout.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        value.contiguous().data_ptr<scalar_t>(),
        dw.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_out, w_out, h_v, w_v);
  });

  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ga_map_backward_kernel_v", [&] {
      ga_map_backward_kernel_v<scalar_t><<<GET_BLOCKS(cal_size_v), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_v,
        dout.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        value.contiguous().data_ptr<scalar_t>(),
        dv.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_out, w_out, h_v, w_v);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(dw, dv);
}

}  // namespace detr

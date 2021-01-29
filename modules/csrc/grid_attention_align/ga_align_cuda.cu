#include "common_cuda_helper.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

template <typename T>
__global__ void ga_align_forward_kernel(
    const int nthreads, const T* query, const T* key, T* weight,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_q, const int w_q,
    const int h_k, const int w_k) {
  int sp_q = h_q * w_q;
  int sp_k = h_k * w_k;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_k) / static_cast<T>(w_q);
  T step_h = static_cast<T>(h_k) / static_cast<T>(h_q);
  int bin_grid_w = static_cast<int>(ceil(w_k / w_q));
  int bin_grid_h = static_cast<int>(ceil(h_k / h_q));
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_q;
    int gh = (index / w_q) % h_q;
    int c = (index / w_q / h_q) % len;
    int n = index / w_q / h_q / len;

    T start_w = static_cast<T>(gw) * step_w;
    T start_h = static_cast<T>(gh) * step_h;

    int bin_x_idx = c % local_attn_w;
    int bin_y_idx = (c / local_attn_w) % local_attn_h;

    const T x = start_w + static_cast<T>(bin_x_idx);
    const T y = start_h + static_cast<T>(bin_y_idx);

    for (int plane = 0; plane < channels; ++plane) {
      T _q = query[(n * channels + plane) * sp_q + gh * w_q + gw];
      const T* offset_key = key + (n * channels + plane) * sp_k;
      T _k_inter = bilinear_interpolate(offset_key, h_k, w_k, y, x, index);
      weight[(n * len + c) * sp_q + gh * w_q + gw] += _q * _k_inter;
    }
  }
}

template <typename T>
__global__ void ga_align_backward_kernel_q(
    const int nthreads, const T* dw, const T* query, const T* key, T* dq,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_q, const int w_q,
    const int h_k, const int w_k) {
  int sp_q = h_q * w_q;
  int sp_k = h_k * w_k;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_k) / static_cast<T>(w_q);
  T step_h = static_cast<T>(h_k) / static_cast<T>(w_k);
  int bin_grid_w = static_cast<int>(ceil(w_k / w_q));
  int bin_grid_h = static_cast<int>(ceil(h_k / h_q));
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_q;
    int gh = (index / w_q) % h_q;
    int c = (index / w_q / h_q) % channels;
    int n = index / w_q / h_q / channels;

    T start_w = static_cast<T>(gw) * step_w;
    T start_h = static_cast<T>(gh) * step_h;

    const T* offset_key = key + (n * channels + c) * sp_k;

    for (int h = 0; h < local_attn_h; ++h) {
      for (int w = 0; w < local_attn_w; ++w) {
        T _dw = dw[(n * len + h * local_attn_w + w) * sp_q + gh * w_q + gw];
        T x = start_w + static_cast<T>(w);
        T y = start_h + static_cast<T>(h);
        x = min(max(x, T(0.)), T(w_k));
        y = min(max(y, T(0.)), T(h_k));
        T _k_inter = bilinear_interpolate(offset_key, h_k, w_k, y, x, index);
        dq[(n * channels + c) * sp_q + gh * w_q + gw] += _dw * _k_inter;
      }
    }
  }
}

template <typename T>
__global__ void ga_align_backward_kernel_k(
    const int nthreads, const T* dw, const T* query, const T* key, T* dk,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_q, const int w_q,
    const int h_k, const int w_k) {
  int sp_q = h_q * w_q;
  int sp_k = h_k * w_k;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_k) / static_cast<T>(w_q);
  T step_h = static_cast<T>(h_k) / static_cast<T>(h_q);
  int bin_grid_w = static_cast<int>(ceil(w_k / w_q));
  int bin_grid_h = static_cast<int>(ceil(h_k / h_q));
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, kw, kh) is an element in the output
    int qw = index % w_q;
    int qh = (index / w_q) % h_q;
    int c = (index / w_q / h_q) % channels;
    int n = index / w_q / h_q / channels;

    T start_w = static_cast<T>(qw) * step_w;
    T start_h = static_cast<T>(qh) * step_h;

    const T _q = query[(n * channels + c) * sp_q + qh * w_q + qw];
    T* offset_dk = dk + (n * channels + c) * sp_k;

    for (int h = 0; h < local_attn_h; ++h) {
      for (int w = 0; w < local_attn_w; ++w) {
        T _dw = dw[(n * len + h * local_attn_w + w) * sp_q + qh * w_q + qw];
        T x = start_w + static_cast<T>(w);
        T y = start_h + static_cast<T>(h);
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(h_k, w_k, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_dk + y_low * w_k + x_low,
                    _dw * w1 * _q);
          atomicAdd(offset_dk + y_low * w_k + x_high,
                    _dw * w2 * _q);
          atomicAdd(offset_dk + y_high * w_k + x_low,
                    _dw * w3 * _q);
          atomicAdd(offset_dk + y_high * w_k + x_high,
                    _dw * w4 * _q);
        }
      }
    }
  }
}

template <typename T>
__global__ void ga_map_align_forward_kernel(
    const int nthreads, const T* weight, const T* value, T* out,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_w, const int w_w,
    const int h_v, const int w_v) {
  int sp_w = h_w * w_w;
  int sp_v = h_v * w_v;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_v) / static_cast<T>(w_w);
  T step_h = static_cast<T>(h_v) / static_cast<T>(h_w);
  int bin_grid_w = static_cast<int>(ceil(w_v / w_w));
  int bin_grid_h = static_cast<int>(ceil(h_v / h_w));
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_w;
    int gh = (index / w_w) % h_w;
    int c = (index / w_w / h_w) % channels;
    int n = index / w_w / h_w / channels;

    T start_w = static_cast<T>(gw) * step_w;
    T start_h = static_cast<T>(gh) * step_h;

    const T* offset_val = value + (n* channels + c) * sp_v;

    for (int h = 0; h < local_attn_h; ++h) {
      for (int w = 0; w < local_attn_w; ++w) {
        T _w = weight[(n * len + h * local_attn_w + w) * sp_w + gh * w_w + gw];
        const T x = start_w + static_cast<T>(w);
        const T y = start_h + static_cast<T>(h);
        T _v_inter = bilinear_interpolate(offset_val, h_v, w_v, y, x, index);
        out[(n * channels + c) * sp_w + gh * w_w + gw] = _w * _v_inter;
      }
    }
  }
}

template <typename T>
__global__ void ga_map_align_backward_kernel_w(
    const int nthreads, const T* dout, const T* weight, const T* value, T* dw,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_out, const int w_out,
    const int h_v, const int w_v) {
  int sp_out = h_out * w_out;
  int sp_v = h_v * w_v;
  int len = local_attn_h * local_attn_w;
  T step_h = static_cast<T>(w_v) / static_cast<T>(w_out);
  T step_w = static_cast<T>(h_v) / static_cast<T>(h_out);
  int bin_grid_w = static_cast<int>(ceil(w_v / w_out));
  int bin_grid_h = static_cast<int>(ceil(h_v / h_out));
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, gh, gw) is an element in the output
    int gw = index % w_out;
    int gh = (index / w_out) % h_out;
    int c = (index / w_out / h_out) % len;
    int n = index / w_out / h_out / len;

    T start_w = static_cast<T>(gw) * step_w;
    T start_h = static_cast<T>(gh) * step_h;

    int bin_x_idx = c % local_attn_w;
    int bin_y_idx = (c / local_attn_w) % local_attn_h;

    const T x = start_w + static_cast<T>(bin_x_idx);
    const T y = start_h + static_cast<T>(bin_y_idx);

    for (int plane = 0; plane < channels; ++plane) {
      T _dout = dout[(n * channels + plane) * sp_out + gh * w_out + gw];
      const T* offset_val = value + (n * channels + plane) * sp_v;
      T _v_inter = bilinear_interpolate(offset_val, h_v, w_v, y, x, index);
      dw[(n * len + c) * sp_out + gh * w_out + gw] += _dout * _v_inter;
    }
  }
}

template <typename T>
__global__ void ga_map_align_backward_kernel_v(
    const int nthreads, const T* dout, const T* weight, const T* value, T* dv,
    const int local_attn_h, const int local_attn_w, const int num,
    const int channels, const int h_out, const int w_out,
    const int h_v, const int w_v) {
  int sp_out = h_out * w_out;
  int sp_v = h_v * w_v;
  int len = local_attn_h * local_attn_w;
  T step_w = static_cast<T>(w_v) / static_cast<T>(w_out);
  T step_h = static_cast<T>(h_v) / static_cast<T>(h_out);
  int bin_grid_w = static_cast<int>(ceil(w_v / w_out));
  int bin_grid_h = static_cast<int>(ceil(h_v / h_out));
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, vh, vw) is an element in the output
    int outw = index % w_out;
    int outh = (index / w_out) % h_out;
    int c = (index / w_out / h_out) % channels;
    int n = index / w_out / h_out / channels;

    T start_w = static_cast<T>(outw) * step_w;
    T start_h = static_cast<T>(outh) * step_h;

    const T _dout = dout[(n * channels + c) * sp_out + outh * w_out + outw];
    T* offset_dv = dv + (n * channels + c) * sp_v;

    for (int h = 0; h < local_attn_h; ++h) {
      for (int w = 0; w < local_attn_w; ++w) {
        const T _w = weight[(n * len + h * local_attn_w + w) * sp_out + outh * w_out + outw];
        T x = start_w + static_cast<T>(w);
        T y = start_h + static_cast<T>(h);
        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(h_v, w_v, y, x, w1, w2, w3, w4,
                                      x_low, x_high, y_low, y_high, index);
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_dv + y_low * w_v + x_low,
                    _dout * w1 * _w);
          atomicAdd(offset_dv + y_low * w_v + x_high,
                    _dout * w2 * _w);
          atomicAdd(offset_dv + y_high * w_v + x_low,
                    _dout * w3 * _w);
          atomicAdd(offset_dv + y_high * w_v + x_high,
                    _dout * w4 * _w);
        }
      }
    }
  }
}


namespace detr {

at::Tensor ga_align_forward_cuda(const at::Tensor& query,
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

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "ga_align_forward", [&] {
      ga_align_forward_kernel<scalar_t><<<GET_BLOCKS(cal_size), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size,
        query.contiguous().data_ptr<scalar_t>(),
        key.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_q, w_q, h_k, w_k);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return weight;
}

std::tuple<at::Tensor, at::Tensor> ga_align_backward_cuda(const at::Tensor& dw,
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
  int cal_size_k = dq.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "ga_align_backward_kernel_q", [&] {
      ga_align_backward_kernel_q<scalar_t><<<GET_BLOCKS(cal_size_q), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_q,
        dw.contiguous().data_ptr<scalar_t>(),
        query.contiguous().data_ptr<scalar_t>(),
        key.contiguous().data_ptr<scalar_t>(),
        dq.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_q, w_q, h_k, w_k);
  });

  AT_DISPATCH_FLOATING_TYPES(key.scalar_type(), "ga_align_backward_kernel_k", [&] {
      ga_align_backward_kernel_k<scalar_t><<<GET_BLOCKS(cal_size_k), THREADS_PER_BLOCK, 0, stream>>>(
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

at::Tensor ga_map_align_forward_cuda(const at::Tensor& weight,
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

  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ga_map_align_forward_kernel", [&] {
      ga_map_align_forward_kernel<scalar_t><<<GET_BLOCKS(cal_size_out), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_out,
        weight.contiguous().data_ptr<scalar_t>(),
        value.contiguous().data_ptr<scalar_t>(),
        out.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_w, w_w, h_v, w_v);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

std::tuple<at::Tensor, at::Tensor> ga_map_align_backward_cuda(const at::Tensor& dout,
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
  int cal_size_v = dw.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "ga_map_align_backward_kernel_w", [&] {
      ga_map_align_backward_kernel_w<scalar_t><<<GET_BLOCKS(cal_size_w), THREADS_PER_BLOCK, 0, stream>>>(
        cal_size_w,
        dout.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        value.contiguous().data_ptr<scalar_t>(),
        dw.contiguous().data_ptr<scalar_t>(),
        local_attn_h, local_attn_w, n, c, h_out, w_out, h_v, w_v);
  });

  AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ga_map_align_backward_kernel_v", [&] {
      ga_map_align_backward_kernel_v<scalar_t><<<GET_BLOCKS(cal_size_v), THREADS_PER_BLOCK, 0, stream>>>(
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

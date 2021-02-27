#include "common_cuda_helper.hpp"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


inline int GET_BLOCKS_CUSTOM(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


template <typename scalar_t>
__device__ scalar_t cc_bilinear(const scalar_t* input, const int height,
                                const int width, scalar_t y, scalar_t x,
                                const int index) {
  int y_low = floor(y);
  int x_low = floor(x);
  int y_high = y_low + 1;
  int x_high = y_high + 1;

  const scalar_t ly = y - y_low;
  const scalar_t lx = x - x_low;
  const scalar_t hy = 1 - ly, hx = 1 - lx;
  // do bilinear interpolation
  scalar_t v1 = 0;
  //if (y_low >= 0 && x_low >= 0)
  //{
  v1 = input[y_low * width + x_low];
  //}
  scalar_t v2 = 0;
  //if (y_low >= 0 && x_high <= width - 1)
  //{
  v2 = input[y_low * width + x_high];
  //}
  scalar_t v3 = 0;
  //if (y_high <= height - 1 && x_low >= 0)
  //{
  v3 = input[y_high * width + x_low];
  //}
  scalar_t v4 = 0;
  //if (y_high <= height - 1 && x_high <= width - 1)
  //{
  v4 = input[y_high * width + x_high];
  //}
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}


template <typename scalar_t>
__device__ void cc_bilinear_gradient(const scalar_t* &bottom_data, 
                                     const int &height, const int &width,
                                     const scalar_t &h, const scalar_t &w,
                                     const scalar_t &top_grad,
                                     const scalar_t &attn_weight,
                                     scalar_t* &grad_value,
                                     scalar_t* grad_weight,
                                     scalar_t* grad_point)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    v1 = bottom_data[h_low * width + w_low];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + h_low * width + w_low, w1*top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    v2 = bottom_data[h_low * width + w_high];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + h_low * width + w_high, w2*top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    v3 = bottom_data[h_high * width + w_low];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + h_high * width + w_low, w3*top_grad_value); 
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    v4 = bottom_data[h_high * width + w_high];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + h_high * width + w_high, w4*top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_weight = top_grad * val;
  *grad_point = width * grad_w_weight * top_grad_value;
  *(grad_point + 1) = height * grad_h_weight * top_grad_value;
}


template <typename scalar_t>
__global__ void cross_ca_weight_forward_kernel(const int n,
                                               const scalar_t *query,
                                               const scalar_t *key,
                                               const scalar_t *point,
                                               scalar_t *output,
                                               const int batch_size,
                                               const int num_heads,
                                               const int channels,
                                               const int height,
                                               const int width,
                                               const int num_query) {
  const int len_weight = (height + width - 1);
  const int sp_offset = height * width;
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int weight_col = index % len_weight;
    const int head_col = (index / len_weight) % num_heads;
    const int query_col = (index / len_weight / num_heads) % num_query;
    const int batch_col = (index / len_weight / num_heads / num_query) % batch_size;

    scalar_t *output_ptr = output + index;
    int point_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * 2;
    int key_base_ptr = batch_col * num_heads * channels * sp_offset + head_col * channels * sp_offset;
    int query_base_ptr = batch_col * num_query * num_heads * channels + query_col * num_heads * channels + head_col * channels;

    const scalar_t point_x = point[point_ptr] * width;
    const scalar_t point_y = point[point_ptr + 1] * height;
    const scalar_t start_x = point_x - floor(point_x);
    const scalar_t start_y = point_y - floor(point_y);
    scalar_t x = 0;
    scalar_t y = 0;

    if (weight_col < width) {
      x = start_x + weight_col - 0.5;
      y = point_y - 0.5;
    } else {
      int h_index = weight_col - width;
      y = start_y + h_index;
      int j = y < point_y ? h_index : h_index + 1;
      x = point_x - 0.5;
      y = start_y + j - 0.5;
    }
    scalar_t col = 0;
    for (int plane = 0; plane < channels; ++plane) {
      const scalar_t* offset_key = key + key_base_ptr + plane * sp_offset;
      scalar_t key_inter = bilinear_interpolate(offset_key, height, width, y, x, index);
      scalar_t query_inter = query[query_base_ptr + plane];
      col += key_inter * query_inter;
    }
    *output_ptr = col;
  }
}


template <typename scalar_t>
__global__ void cross_ca_weight_backward_kernel_v1(const int n,
                                                   const scalar_t *grad_w,
                                                   const scalar_t *query,
                                                   const scalar_t *key,
                                                   const scalar_t *point,
                                                   scalar_t *grad_query,
                                                   scalar_t *grad_key,
                                                   scalar_t *grad_point,
                                                   const int batch_size,
                                                   const int num_heads,
                                                   const int channels,
                                                   const int height,
                                                   const int width,
                                                   const int num_query) {
  const int len_weight = (height + width - 1);
  const int sp_offset = height * width;
  CUDA_1D_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    scalar_t* cache_grad_points = (scalar_t*)_s;
    scalar_t* cache_grad_query = cache_grad_points + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    const int weight_col = index % len_weight;
    const int head_col = (index / len_weight) % num_heads;
    const int query_col = (index / len_weight / num_heads) % num_query;
    const int batch_col = (index / len_weight / num_heads / num_query) % batch_size;

    const scalar_t top_grad = grad_w[index];
    int point_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * 2;
    int key_base_ptr = batch_col * num_heads * channels * sp_offset + head_col * channels * sp_offset;
    int query_base_ptr = batch_col * num_query * num_heads * channels + query_col * num_heads * channels + head_col * channels;

    const scalar_t point_x = point[point_ptr] * width;
    const scalar_t point_y = point[point_ptr + 1] * height;
    const scalar_t start_x = point_x - floor(point_x);
    const scalar_t start_y = point_y - floor(point_y);
    scalar_t x = 0;
    scalar_t y = 0;

    if (weight_col < width) {
      x = start_x + weight_col - 0.5;
      y = point_y - 0.5;
    } else {
      int h_index = weight_col - width;
      y = start_y + h_index;
      int j = y < point_y ? h_index : h_index + 1;
      x = point_x - 0.5;
      y = start_y + j - 0.5;
    }
    scalar_t grad_point_w = 0;
    scalar_t grad_point_h = 0;
    for (int plane = 0; plane < channels; ++plane) {
      const int offset_key = key_base_ptr + plane * sp_offset;
      const int offset_query = query_base_ptr + plane;
      const scalar_t *key_ptr = key + offset_key;
      scalar_t *grad_key_ptr = grad_key + offset_key;
      const scalar_t query_val = query[offset_query];
      *(cache_grad_points+(threadIdx.x << 1)) = 0;
      *(cache_grad_points+((threadIdx.x << 1) + 1)) = 0;
      *(cache_grad_query + threadIdx.x) = 0;
      if (y > -1 && x > -1 && y < height && x < width) {
        cc_bilinear_gradient(key_ptr, height, width, y, x, top_grad, query_val,
                             grad_key_ptr, cache_grad_query + threadIdx.x, cache_grad_points + (threadIdx.x << 1));
      }
      __syncthreads();

      if (tid == 0) {
        scalar_t _grad_w = cache_grad_points[0], _grad_h = cache_grad_points[1], _grad_a = cache_grad_query[0];
        int sid = 2;
        for (unsigned int tid = 1; tid < blockDim.x; ++tid) {
          _grad_a += cache_grad_query[tid];
          _grad_w += cache_grad_points[sid];
          _grad_h += cache_grad_points[sid + 1];
          sid += 2;
        }

        *(grad_query + offset_query) = _grad_a;
        grad_point_w += _grad_w;
        grad_point_h += _grad_h;
      }
      __syncthreads();
    }
    *(grad_point + point_ptr) = grad_point_w;
    *(grad_point + point_ptr + 1) = grad_point_h;
  }
}


template <typename scalar_t>
__global__ void cross_ca_weight_backward_kernel_v2(const int n,
                                                   const scalar_t *grad_w,
                                                   const scalar_t *query,
                                                   const scalar_t *key,
                                                   const scalar_t *point,
                                                   scalar_t *grad_query,
                                                   scalar_t *grad_key,
                                                   scalar_t *grad_point,
                                                   const int batch_size,
                                                   const int num_heads,
                                                   const int channels,
                                                   const int height,
                                                   const int width,
                                                   const int num_query) {
  const int len_weight = (height + width - 1);
  const int sp_offset = height * width;
  CUDA_1D_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    scalar_t* cache_grad_points = (scalar_t*)_s;
    scalar_t* cache_grad_query = cache_grad_points + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    const int weight_col = index % len_weight;
    const int head_col = (index / len_weight) % num_heads;
    const int query_col = (index / len_weight / num_heads) % num_query;
    const int batch_col = (index / len_weight / num_heads / num_query) % batch_size;

    const scalar_t top_grad = grad_w[index];
    int point_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * 2;
    int key_base_ptr = batch_col * num_heads * channels * sp_offset + head_col * channels * sp_offset;
    int query_base_ptr = batch_col * num_query * num_heads * channels + query_col * num_heads * channels + head_col * channels;

    const scalar_t point_x = point[point_ptr] * width;
    const scalar_t point_y = point[point_ptr + 1] * height;
    const scalar_t start_x = point_x - floor(point_x);
    const scalar_t start_y = point_y - floor(point_y);
    scalar_t x = 0;
    scalar_t y = 0;

    if (weight_col < width) {
      x = start_x + weight_col - 0.5;
      y = point_y - 0.5;
    } else {
      int h_index = weight_col - width;
      y = start_y + h_index;
      int j = y < point_y ? h_index : h_index + 1;
      x = point_x - 0.5;
      y = start_y + j - 0.5;
    }
    scalar_t grad_point_w = 0;
    scalar_t grad_point_h = 0;
    for (int plane = 0; plane < channels; ++plane) {
      const int offset_key = key_base_ptr + plane * sp_offset;
      const int offset_query = query_base_ptr + plane;
      const scalar_t *key_ptr = key + offset_key;
      scalar_t *grad_key_ptr = grad_key + offset_key;
      const scalar_t query_val = query[offset_query];
      *(cache_grad_points+(threadIdx.x << 1)) = 0;
      *(cache_grad_points+((threadIdx.x << 1) + 1)) = 0;
      *(cache_grad_query + threadIdx.x) = 0;
      if (y > -1 && x > -1 && y < height && x < width) {
        cc_bilinear_gradient(key_ptr, height, width, y, x, top_grad, query_val,
                             grad_key_ptr, cache_grad_query + threadIdx.x, cache_grad_points + (threadIdx.x << 1));
      }
      __syncthreads();

      for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1) {
        if (tid < s) {
          const unsigned int xid1 = tid << 1;
          const unsigned int xid2 = (tid + s) << 1;
          cache_grad_query[tid] += cache_grad_query[tid + s];
          cache_grad_points[xid1] += cache_grad_points[xid2];
          cache_grad_points[xid1 + 1] += cache_grad_points[xid2 + 1];
          if (tid + (s << 1) < spre)
          {
            cache_grad_query[tid] += cache_grad_query[tid + (s << 1)];
            cache_grad_points[xid1] += cache_grad_points[xid2 + (s << 1)];
            cache_grad_points[xid1 + 1] += cache_grad_points[xid2 + 1 + (s << 1)];
          } 
        }
        __syncthreads();
      }

      if (tid == 0) {
        *(grad_query + offset_query) = cache_grad_query[0];
        grad_point_w += cache_grad_points[0];
        grad_point_h += cache_grad_points[1];
      }
      __syncthreads();
    }
    *(grad_point + point_ptr) = grad_point_w;
    *(grad_point + point_ptr + 1) = grad_point_h;
  }
}


template <typename scalar_t>
__global__ void cross_ca_map_forward_kernel(const int n,
                                            const scalar_t *weight,
                                            const scalar_t *value,
                                            const scalar_t *point,
                                            scalar_t *output,
                                            const int batch_size,
                                            const int num_heads,
                                            const int channels,
                                            const int height,
                                            const int width,
                                            const int num_query) {
  const int len_weight = (height + width - 1);
  const int sp_offset = height * width;
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int c_col = index % channels;
    const int head_col = (index / channels) % num_heads;
    const int query_col = (index / channels / num_heads) % num_query;
    const int batch_col = (index / channels / num_heads / num_query) % batch_size;

    scalar_t *output_ptr = output + index;
    int weight_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * len_weight;
    int point_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * 2;
    const int data_value_ptr_init_offset = batch_col  * num_heads * channels * sp_offset + head_col * channels * sp_offset + c_col * sp_offset;
    const scalar_t *data_value_ptr = value + data_value_ptr_init_offset;
    scalar_t col = 0;

    const scalar_t point_x = point[point_ptr] * width;
    const scalar_t point_y = point[point_ptr + 1] * height;
    const scalar_t start_x = point_x - floor(point_x);
    const scalar_t start_y = point_y - floor(point_y);

    for (int i = 0; i < width; ++i) {
      scalar_t x = start_x + i - 0.5;
      scalar_t y = point_y - 0.5;
      const scalar_t point_weight = weight[weight_ptr + i];
      if (y > -1 && x > -1 && y < height && x < width) {
        col += bilinear_interpolate(data_value_ptr, height, width, y, x, index) * point_weight;
      }
    }
    for (int i = 0; i < height; ++i) {
      if ((start_y + i) == point_y) continue;

      int j = (start_y + i) < point_y ? i : i - 1;
      scalar_t x = point_x - 0.5;
      scalar_t y = start_y + i - 0.5;
      const scalar_t point_weight = weight[weight_ptr + width + j];
      if (y > -1 && x > -1 && y < height && x < width) {
        col += bilinear_interpolate(data_value_ptr, height, width, y, x, index) * point_weight;
      }
    }
    *output_ptr = col;
  }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void cross_ca_map_backward_kernel_v1(const int n,
                                                const scalar_t *grad_out,
                                                const scalar_t *weight,
                                                const scalar_t *value,
                                                const scalar_t *point,
                                                scalar_t* grad_weight,
                                                scalar_t* grad_value,
                                                scalar_t* grad_point,
                                                const int batch_size,
                                                const int num_heads,
                                                const int channels,
                                                const int height,
                                                const int width,
                                                const int num_query) {
  const int sp_offset = height * width;
  CUDA_1D_KERNEL_LOOP(index, n) {
    __shared__ scalar_t cache_grad_points[blockSize * 2];
    __shared__ scalar_t cache_grad_weight[blockSize];
    unsigned int tid = threadIdx.x;
    const int c_col = index % channels;
    const int head_col = (index / channels) % num_heads;
    const int query_col = (index / channels / num_heads) % num_query;
    const int batch_col = (index / channels / num_heads / num_query) % batch_size;
    const int sampling_index = index / channels;

    const scalar_t top_grad = grad_out[index];
    int weight_ptr = sampling_index * (height + width - 1);
    int point_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * 2;
    const int data_value_ptr_init_offset = batch_col  * num_heads * channels * sp_offset + head_col * channels * sp_offset + c_col * sp_offset;
    const scalar_t *data_value_ptr = value + data_value_ptr_init_offset;
    scalar_t *grad_value_ptr = grad_value + data_value_ptr_init_offset;
    scalar_t *grad_point_ptr = grad_point + point_ptr;

    const scalar_t point_x = point[point_ptr] * width;
    const scalar_t point_y = point[point_ptr + 1] * height;
    const scalar_t start_x = point_x - floor(point_x);
    const scalar_t start_y = point_y - floor(point_y);

    scalar_t grad_point_w = 0;
    scalar_t grad_point_h = 0;
    for (int i = 0; i < width; ++i) {
      scalar_t x = start_x + i - 0.5;
      scalar_t y = point_y - 0.5;
      const scalar_t point_weight = weight[weight_ptr + i];
      *(cache_grad_points+(threadIdx.x << 1)) = 0;
      *(cache_grad_points+((threadIdx.x << 1) + 1)) = 0;
      *(cache_grad_weight+threadIdx.x)=0;
      if (y > -1 && x > -1 && y < height && x < width) {
        cc_bilinear_gradient(data_value_ptr, height, width, y, x, top_grad, point_weight,
                             grad_value_ptr, cache_grad_weight + threadIdx.x, cache_grad_points + (threadIdx.x << 1));
      }

      __syncthreads();
      if (tid == 0) {
        scalar_t _grad_w = cache_grad_points[0], _grad_h = cache_grad_points[1], _grad_a = cache_grad_weight[0];
        int sid = 2;
        for (unsigned int tid = 1; tid < blockSize; ++tid) {
          _grad_a += cache_grad_weight[tid];
          _grad_w += cache_grad_points[sid];
          _grad_h += cache_grad_points[sid + 1];
          sid += 2;
        }

        *(grad_weight + weight_ptr + i) = _grad_a;
        grad_point_w += _grad_w;
        grad_point_h += _grad_h;
      }
      __syncthreads();
    }

    for (int i = 0; i < height; ++i) {
      if ((start_y + i) == point_y) continue;

      int j = (start_y + i) < point_y ? i : i - 1;
      scalar_t x = point_x - 0.5;
      scalar_t y = start_y + i - 0.5;
      const scalar_t point_weight = weight[weight_ptr + width + j];
      *(cache_grad_points+(threadIdx.x << 1)) = 0;
      *(cache_grad_points+((threadIdx.x << 1) + 1)) = 0;
      *(cache_grad_weight+threadIdx.x)=0;
      if (y > -1 && x > -1 && y < height && x < width) {
        cc_bilinear_gradient(data_value_ptr, height, width, y, x, top_grad, point_weight,
                             grad_value_ptr, cache_grad_weight + threadIdx.x, cache_grad_points + (threadIdx.x << 1));
      }
      
      __syncthreads();
      if (tid == 0) {
        scalar_t _grad_w = cache_grad_points[0], _grad_h = cache_grad_points[1], _grad_a = cache_grad_weight[0];
        int sid = 2;
        for (unsigned int tid = 1; tid < blockSize; ++tid) {
          _grad_a += cache_grad_weight[tid];
          _grad_w += cache_grad_points[sid];
          _grad_h += cache_grad_points[sid + 1];
          sid += 2;
        }

        *(grad_weight + weight_ptr + width + j) = _grad_a;
        grad_point_w += _grad_w;
        grad_point_h += _grad_h;
      }
      __syncthreads();
    }
    *(grad_point + point_ptr) = grad_point_w;
    *(grad_point + point_ptr + 1) = grad_point_h;
  }
}


template <typename scalar_t, unsigned int blockSize>
__global__ void cross_ca_map_backward_kernel_v2(const int n,
                                                const scalar_t *grad_out,
                                                const scalar_t *weight,
                                                const scalar_t *value,
                                                const scalar_t *point,
                                                scalar_t* grad_weight,
                                                scalar_t* grad_value,
                                                scalar_t* grad_point,
                                                const int batch_size,
                                                const int num_heads,
                                                const int channels,
                                                const int height,
                                                const int width,
                                                const int num_query) {
  const int sp_offset = height * width;
  CUDA_1D_KERNEL_LOOP(index, n) {
    __shared__ scalar_t cache_grad_points[blockSize * 2];
    __shared__ scalar_t cache_grad_weight[blockSize];
    unsigned int tid = threadIdx.x;
    const int c_col = index % channels;
    const int head_col = (index / channels) % num_heads;
    const int query_col = (index / channels / num_heads) % num_query;
    const int batch_col = (index / channels / num_heads / num_query) % batch_size;
    const int sampling_index = index / channels;

    const scalar_t top_grad = grad_out[index];
    int weight_ptr = sampling_index * (height + width - 1);
    int point_ptr = (batch_col * num_query * num_heads + query_col * num_heads + head_col) * 2;
    const int data_value_ptr_init_offset = batch_col  * num_heads * channels * sp_offset + head_col * channels * sp_offset + c_col * sp_offset;
    const scalar_t *data_value_ptr = value + data_value_ptr_init_offset;
    scalar_t *grad_value_ptr = grad_value + data_value_ptr_init_offset;
    scalar_t *grad_point_ptr = grad_point + point_ptr;

    const scalar_t point_x = point[point_ptr] * width;
    const scalar_t point_y = point[point_ptr + 1] * height;
    const scalar_t start_x = point_x - floor(point_x);
    const scalar_t start_y = point_y - floor(point_y);

    scalar_t grad_point_w = 0;
    scalar_t grad_point_h = 0;
    for (int i = 0; i < width; ++i) {
      scalar_t x = start_x + i - 0.5;
      scalar_t y = point_y - 0.5;
      const scalar_t point_weight = weight[weight_ptr + i];
      *(cache_grad_points+(threadIdx.x << 1)) = 0;
      *(cache_grad_points+((threadIdx.x << 1) + 1)) = 0;
      *(cache_grad_weight+threadIdx.x)=0;
      if (y > -1 && x > -1 && y < height && x < width) {
        cc_bilinear_gradient(data_value_ptr, height, width, y, x, top_grad, point_weight,
                             grad_value_ptr, cache_grad_weight + threadIdx.x, cache_grad_points + (threadIdx.x << 1));
      }

      __syncthreads();

      for (unsigned int s=blockSize/2; s > 0; s>>=1) {
        if (tid < s) {
          const unsigned int xid1 = tid << 1;
          const unsigned int xid2 = (tid + s) << 1;
          cache_grad_weight[tid] += cache_grad_weight[tid + s];
          cache_grad_points[xid1] += cache_grad_points[xid2];
          cache_grad_points[xid1 + 1] += cache_grad_points[xid2 + 1];
        }
        __syncthreads();
      }

      if (tid == 0) {
        *(grad_weight + weight_ptr + i) = cache_grad_weight[0];
        grad_point_w += cache_grad_points[0];
        grad_point_h += cache_grad_points[1];
      }
      __syncthreads();
    }

    for (int i = 0; i < height; ++i) {
      if ((start_y + i) == point_y) continue;

      int j = (start_y + i) < point_y ? i : i - 1;
      scalar_t x = point_x - 0.5;
      scalar_t y = start_y + i - 0.5;
      const scalar_t point_weight = weight[weight_ptr + width + j];
      *(cache_grad_points+(threadIdx.x << 1)) = 0;
      *(cache_grad_points+((threadIdx.x << 1) + 1)) = 0;
      *(cache_grad_weight+threadIdx.x)=0;
      if (y > -1 && x > -1 && y < height && x < width) {
        cc_bilinear_gradient(data_value_ptr, height, width, y, x, top_grad, point_weight,
                             grad_value_ptr, cache_grad_weight + threadIdx.x, cache_grad_points + (threadIdx.x << 1));
      }
      
      __syncthreads();

      for (unsigned int s=blockSize/2; s > 0; s>>=1) {
        if (tid < s) {
          const unsigned int xid1 = tid << 1;
          const unsigned int xid2 = (tid + s) << 1;
          cache_grad_weight[tid] += cache_grad_weight[tid + s];
          cache_grad_points[xid1] += cache_grad_points[xid2];
          cache_grad_points[xid1 + 1] += cache_grad_points[xid2 + 1];
        }
        __syncthreads();
      }

      if (tid == 0) {
        *(grad_weight + weight_ptr + width + j) = cache_grad_weight[0];
        grad_point_w += cache_grad_points[0];
        grad_point_h += cache_grad_points[1];
      }
      __syncthreads();
    }
    *(grad_point + point_ptr) = grad_point_w;
    *(grad_point + point_ptr + 1) = grad_point_h;
  }
}


namespace detr {

at::Tensor cross_ca_weight_forward_cuda(const at::Tensor& query,
                                        const at::Tensor& key,
                                        const at::Tensor& point) {
    AT_ASSERTM(query.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(key.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(point.device().is_cuda(), "input must be a CUDA tensor");

    const int batch_size = key.size(0);
    const int num_heads = key.size(1);
    const int channels = key.size(2);
    const int height = key.size(3);
    const int width = key.size(4);

    const int num_query = point.size(1);

    auto output = at::zeros({batch_size, num_query, num_heads, height + width - 1}, query.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int num_kernels = batch_size * num_query * num_heads * (height + width - 1);

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cross_ca_weight_forward", [&] {
        cross_ca_weight_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
          num_kernels,
          query.contiguous().data_ptr<scalar_t>(),  // (b, num_query, num_heads, channels)
          key.contiguous().data_ptr<scalar_t>(),  // (b, num_heads, channels, height, width)
          point.contiguous().data_ptr<scalar_t>(),  // (b, num_query, num_heads, 2)
          output.contiguous().data_ptr<scalar_t>(),  // (b, num_query, num_heads, height + width - 1)
          batch_size, num_heads, channels, height, width, num_query);
    });

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cross_ca_weight_backward_cuda(const at::Tensor& dw,
                                                                             const at::Tensor& query,
                                                                             const at::Tensor& key,
                                                                             const at::Tensor& point) {
    AT_ASSERTM(dw.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(query.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(key.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(point.device().is_cuda(), "input must be a CUDA tensor");

    const int batch_size = key.size(0);
    const int num_heads = key.size(1);
    const int channels = key.size(2);
    const int height = key.size(3);
    const int width = key.size(4);

    const int num_query = point.size(1);

    auto grad_query = at::zeros_like(query);
    auto grad_key = at::zeros_like(key);
    auto grad_point = at::zeros_like(point);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int num_kernels = batch_size * num_query * num_heads * (height + width - 1);
    const int num_threads = height + width - 1;

    if (num_threads < 64) {
      AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cross_ca_weight_backward", [&] {
        cross_ca_weight_backward_kernel_v1<scalar_t>
        <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, num_threads*3*sizeof(scalar_t), stream>>>(
          num_kernels,
          dw.contiguous().data_ptr<scalar_t>(),
          query.contiguous().data_ptr<scalar_t>(),
          key.contiguous().data_ptr<scalar_t>(),
          point.contiguous().data_ptr<scalar_t>(),
          grad_query.contiguous().data_ptr<scalar_t>(),
          grad_key.contiguous().data_ptr<scalar_t>(),
          grad_point.contiguous().data_ptr<scalar_t>(),
          batch_size, num_heads, channels, height, width, num_query);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "cross_ca_weight_backward", [&] {
        cross_ca_weight_backward_kernel_v2<scalar_t>
        <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, num_threads*3*sizeof(scalar_t), stream>>>(
          num_kernels,
          dw.contiguous().data_ptr<scalar_t>(),
          query.contiguous().data_ptr<scalar_t>(),
          key.contiguous().data_ptr<scalar_t>(),
          point.contiguous().data_ptr<scalar_t>(),
          grad_query.contiguous().data_ptr<scalar_t>(),
          grad_key.contiguous().data_ptr<scalar_t>(),
          grad_point.contiguous().data_ptr<scalar_t>(),
          batch_size, num_heads, channels, height, width, num_query);
      });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_query, grad_key, grad_point);
}

at::Tensor cross_ca_map_forward_cuda(const at::Tensor& weight,
                                     const at::Tensor& value,
                                     const at::Tensor& point) {
    AT_ASSERTM(weight.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(value.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(point.device().is_cuda(), "input must be a CUDA tensor");

    const int batch_size = value.size(0);
    const int num_heads = value.size(1);
    const int channels = value.size(2);
    const int height = value.size(3);
    const int width = value.size(4);

    const int num_query = point.size(1);

    auto output = at::zeros({batch_size, num_query, num_heads, channels}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int num_kernels = batch_size * num_query * num_heads * channels;

    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_forward", [&] {
        cross_ca_map_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
          num_kernels,
          weight.contiguous().data_ptr<scalar_t>(),  // (b, num_query, num_heads, height + width - 1)
          value.contiguous().data_ptr<scalar_t>(),  // (b, num_heads, channels, height, width)
          point.contiguous().data_ptr<scalar_t>(),  // (b, num_query, num_heads, 2)
          output.contiguous().data_ptr<scalar_t>(),  // (b, num_query, num_heads, channels)
          batch_size, num_heads, channels, height, width, num_query);
    });

    output = output.view({batch_size, num_query, num_heads*channels});
    AT_CUDA_CHECK(cudaGetLastError());

    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cross_ca_map_backward_cuda(const at::Tensor& dout,
                                                                          const at::Tensor& weight,
                                                                          const at::Tensor& value,
                                                                          const at::Tensor& point) {
    AT_ASSERTM(dout.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(value.device().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(point.device().is_cuda(), "input must be a CUDA tensor");

    const int batch_size = value.size(0);
    const int num_heads = value.size(1);
    const int channels = value.size(2);
    const int height = value.size(3);
    const int width = value.size(4);

    const int num_query = point.size(1);

    auto grad_value = at::zeros_like(value);
    auto grad_weight = at::zeros_like(weight);
    auto grad_point = at::zeros_like(point);
    auto grad_out = dout.view({batch_size, num_query, num_heads, channels});
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = channels;

    switch(channels) {
      case 1:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v1<scalar_t, 1>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 2:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v1<scalar_t, 2>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 4:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v1<scalar_t, 4>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 8:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v1<scalar_t, 8>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 16:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v1<scalar_t, 16>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 32:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v1<scalar_t, 32>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 64:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v2<scalar_t, 64>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 128:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v2<scalar_t, 128>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 256:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v2<scalar_t, 256>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
      case 512:
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "cross_ca_map_backward", [&] {
          cross_ca_map_backward_kernel_v2<scalar_t, 512>
          <<<GET_BLOCKS_CUSTOM(num_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels,
            grad_out.contiguous().data_ptr<scalar_t>(),
            weight.contiguous().data_ptr<scalar_t>(),
            value.contiguous().data_ptr<scalar_t>(),
            point.contiguous().data_ptr<scalar_t>(),
            grad_weight.contiguous().data_ptr<scalar_t>(),
            grad_value.contiguous().data_ptr<scalar_t>(),
            grad_point.contiguous().data_ptr<scalar_t>(),
            batch_size, num_heads, channels, height, width, num_query);
        });
        break;
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_value, grad_weight, grad_point);
}

}  // namespace detr

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "image.h"
#include "cuda.h"
}

__device__ float get_pixel_im_kernel(float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

__device__ float bilinear_interpolate_im_kernel(float *image, int w, int h, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_im_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_im_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_im_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_im_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__global__ void resize_kernel(float *src, const int size, const int height_in, const int width_in, const int channels, const int height_out, const int width_out, float *dst) {

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    int x = id % width_out;
    id /= width_out;
    int y = id % height_out;
    id /= height_out;
    int k = id % channels;

    const float src_x = x * width_in / width_out;
    const float src_y = y * height_in / height_out;

    int out_index = x + width_out*(y + height_out*k);
    dst[out_index] = bilinear_interpolate_im_kernel(src, width_in, height_in, src_x, src_y, k);

}


extern "C" void resize_image_gpu(float *im_in, const int height_in, const int width_in, const int channels, const int height_out, const int width_out, float *im_out){
    
    const int size = channels*height_out*width_out;
    size_t k = (size-1) / BLOCK+1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (size-1)/(x*BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    //printf("%ld %ld %ld %ld\n", size, x, y, x*y*BLOCK);
    resize_kernel<<<d, BLOCK>>>(im_in, size, height_in, width_in, channels, height_out, width_out, im_out);
}

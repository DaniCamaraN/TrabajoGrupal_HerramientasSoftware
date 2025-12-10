import numpy as np
from numba import cuda
import cv2


# ----------------------------
# GPU KERNELS
# ----------------------------

@cuda.jit
def kernel_grayscale(src, dst):
    x, y = cuda.grid(2)
    if x >= src.shape[1] or y >= src.shape[0]:
        return

    b = src[y, x, 0]
    g = src[y, x, 1]
    r = src[y, x, 2]
    dst[y, x] = 0.114 * b + 0.587 * g + 0.299 * r


@cuda.jit
def kernel_blur(src, dst):
    x, y = cuda.grid(2)
    if x < 2 or y < 2 or x >= src.shape[1] - 2 or y >= src.shape[0] - 2:
        return

    s = 0
    for i in range(-2, 3):
        for j in range(-2, 3):
            s += src[y + j, x + i]
    dst[y, x] = s // 25


@cuda.jit
def kernel_and(a, b, out):
    x, y = cuda.grid(2)
    if x >= a.shape[1] or y >= a.shape[0]:
        return
    out[y, x] = a[y, x] & b[y, x]


# ----------------------------
# WRAPPERS CPU/GPU
# ----------------------------

def gpu_grayscale(frame):
    h, w = frame.shape[:2]
    d_src = cuda.to_device(frame)
    d_dst = cuda.device_array((h, w), dtype=np.uint8)

    threads = (16, 16)
    blocks = ((w + 15) // 16, (h + 15) // 16)
    kernel_grayscale[blocks, threads](d_src, d_dst)

    return d_dst.copy_to_host()


def gpu_blur(gray):
    h, w = gray.shape[:2]
    d_src = cuda.to_device(gray)
    d_dst = cuda.device_array_like(d_src)

    threads = (16, 16)
    blocks = ((w + 15) // 16, (h + 15) // 16)
    kernel_blur[blocks, threads](d_src, d_dst)

    return d_dst.copy_to_host()


def gpu_and(a, b):
    h, w = a.shape[:2]
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array_like(d_a)

    threads = (16, 16)
    blocks = ((w + 15) // 16, (h + 15) // 16)
    kernel_and[blocks, threads](d_a, d_b, d_out)

    return d_out.copy_to_host()

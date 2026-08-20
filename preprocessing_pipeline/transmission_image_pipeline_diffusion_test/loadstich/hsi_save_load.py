import numpy as np


def load_hsi(arr):
    """
    Loads a 12-bit HSI image stored with 2 12-bit pixels in 3 8-bit bytes. Assumes the input array is of dtype np.uint8. channels must be a multiple of 3.
    Returns the loaded image with dtype np.uint16
    """
    if not arr.dtype == np.uint8:
        raise ValueError("Input array must have dtype np.uint8")
    shape = arr.shape
    if len(shape) == 3:
        img_shape = True
    else:
        img_shape = False

    arr = arr.flatten()
    flat_len = arr.shape[0]

    idx_arr = np.arange(flat_len // 3) * 3
    fst_uint8 = np.uint16(arr[idx_arr])
    mid_uint8 = np.uint16(arr[idx_arr + 1])
    lst_uint8 = np.uint16(arr[idx_arr + 2])

    arr = np.empty(flat_len * 2 // 3, dtype=np.uint16)
    arr[0::2] = (fst_uint8 << 4) + (mid_uint8 >> 4)
    # arr[..., 1::2] = (lst_uint8 << 4) + ((mid_uint8 & 0xF0) >> 4) # This shit is fucking wrong.
    arr[1::2] = (lst_uint8 << 4) + (mid_uint8 & 0xF)

    if img_shape:
        arr = arr.reshape(shape[0], shape[1], -1)

    return arr


def save_hsi(arr):
    """
    Saves a 12-bit HSI image, currently stored as a 16-bit image, as 2 12-bit pixels packed in 3 8-bit bytes. Assumes the input array is of dtype np.uint16. channels must be a multiple of 2.
    """
    if not arr.dtype == np.uint16:
        raise ValueError(f"Input array must have dtype np.uint16 but has dtype {arr.dtype}")
    shape = arr.shape

    idx_arr = np.arange(shape[-1] // 2) * 2
    fst_uint8 = np.uint8(arr[..., idx_arr] >> 4)
    mid_uint8 = np.uint8(np.uint16(arr[..., idx_arr] << 12) >> 8) + np.uint8(
        np.uint16(arr[..., idx_arr + 1] << 12) >> 12
    )
    lst_uint8 = np.uint8(arr[..., idx_arr + 1] >> 4)

    arr = np.empty(shape[:-1] + (shape[-1] // 2 * 3,), dtype=np.uint8)
    arr[..., 0::3] = fst_uint8
    arr[..., 1::3] = mid_uint8
    arr[..., 2::3] = lst_uint8

    return arr

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft2, fftshift
import imageio


def reconstruct_sos_image_hybrid(kspace: np.ndarray) -> np.ndarray:
    """
    Reconstruct from x-ky-kz hybrid domain to x-y-z image domain.
    iFFT should be done along ky-kz only.
    Input shape: (Nx, Ny, Nz, Nc)
    """
    # iFFT along ky (axis=1) and kz (axis=2)
    img_space = np.fft.ifft2(kspace, axes=(1, 2), norm='ortho')  # shape: (Nx, Ny, Nz, Nc)

    # Sum-of-squares across coils
    img_sos = np.sqrt(np.sum(np.abs(img_space)**2, axis=-1))  # shape: (Nx, Ny, Nz)

    # Take central slice in x-direction (readout is already image domain)
    x = img_sos.shape[0] // 2
    slice_img = img_sos[x, :, :]  # shape: (Ny, Nz)

    # Normalize and convert to uint8 for saving
    slice_img /= np.percentile(slice_img, 99)
    slice_uint8 = np.clip(slice_img * 255, 0, 255).astype(np.uint8)

    return slice_uint8


if __name__ == '__main__':
    fileh5 = h5py.File(r"/home/tony_hu/data/Val/e14110s3_P59904.7.h5")
    data = fileh5['kspace']

    real = data[..., :12]
    imag = data[..., 12:]
    kspace = real + 1j * imag  # shape: (256, 218, 170, 12), dtype=complex64

    mag = np.abs(kspace)
    scale = np.percentile(mag, 99.5)  # or max(mag) if safe
    kspace = kspace / scale
    img = reconstruct_sos_image_hybrid(kspace)

    imageio.imwrite("res.png", img)
    print("Saved as res.png")
import os
import cupy as cp
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import matplotlib.pyplot as plt
import h5py

output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

def save_or_load(filename, compute_func):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        print(f'Loading existing {filename}')
        return np.load(path, allow_pickle=True)
    else:
        result = compute_func()
        if isinstance(result, cp.ndarray):
            result = cp.asnumpy(result)  # GPU → CPU
        np.save(path, result)
        print(f'Saved {filename}')
        return result


if __name__ == '__main__':
    # patches:
    # sp.mri.app.JsenseRecon._get_vars = patched_get_vars



    device = sp.Device(0)  # GPU device (use -1 for CPU)
    print(f'Using device: {device}')

    # Step 1: Load and preprocess
    fileh5 = h5py.File(r"/home/tony_hu/data/Val/e14110s3_P59904.7.h5")
    data = fileh5['kspace']

    def step1():
        real = data[..., :12]
        imag = data[..., 12:]
        kspace = real + 1j * imag
        mag = np.abs(kspace)
        scale = np.percentile(mag, 99.5)
        return kspace / scale

    kspace = save_or_load('step1_scaled_kspace.npy', step1)

    # Step 2: Transpose to (nc, nx, ny, nz)
    def step2():
        return np.transpose(kspace, (3, 0, 1, 2))

    kspace = save_or_load('step2_transposed_kspace.npy', step2)

    # Step 3: Move to GPU
    def step3():
        return sp.to_device(kspace, device=device)

    kspace_dev_np = save_or_load('step3_kspace_dev.npy', step3)
    kspace_dev = sp.to_device(kspace_dev_np, device=device)

    # Step 4: IFFT along Ny, Nz
    def step4():
        return sp.ifft(kspace_dev, axes=(2, 3))

    image_space_np = save_or_load('step4_image_space.npy', step4)
    image_space = sp.to_device(image_space_np, device=device)

    # Step 5: ESPIRiT calibration
    def step5():
        return mr.app.EspiritCalib(image_space, calib_width=24, device=device).run()

    calib_np = save_or_load('step5_calib.npy', step5)
    calib = sp.to_device(calib_np, device=device)


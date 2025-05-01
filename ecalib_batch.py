import os
import glob
import cupy as cp
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import h5py
import bart.python.cfl as cfl


# input and output directories
INPUT_DIR = '/home/tony_hu/data/Train'
OUTPUT_DIR = './output/train'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_calibration(image_space, calibration_width: int, device):
    """
    Run ESPIRiT calibration on the image-space data.
    """
    return mr.app.EspiritCalib(
        image_space,
        calib_width=calibration_width,
        device=device
    ).run()

def process_and_save_calibration(input_path: str, output_name: str, device):
    """
    - Load k-space from HDF5
    - Normalize and rearrange dimensions
    - Transform to image space
    - Compute calibration maps
    - Save only the calibration maps in CFL format
    """
    # load raw k-space
    with h5py.File(input_path, 'r') as f:
        raw = f['kspace'][()]

    # combine real and imaginary channels
    real = raw[..., :12]
    imag = raw[..., 12:]
    kspace = real + 1j * imag

    # normalize by 99.5th percentile magnitude
    mag = np.abs(kspace)
    scale = np.percentile(mag, 99.5)
    kspace /= scale

    # reorder to (channels, x, y, z)
    kspace = np.transpose(kspace, (3, 0, 1, 2))

    # move to device (GPU or CPU)
    kspace_dev = sp.to_device(kspace, device=device)

    # inverse FFT on second and third axes
    image_space = sp.ifft(kspace_dev, axes=(2, 3))

    # compute calibration maps
    calib_maps = compute_calibration(image_space, calibration_width=24, device=device)

    # bring to host if needed
    if isinstance(calib_maps, cp.ndarray):
        calib_maps = cp.asnumpy(calib_maps)

    # write out CFL (.cfl + .hdr)
    cfl.writecfl(os.path.join(OUTPUT_DIR, output_name), calib_maps)

if __name__ == '__main__':
    device = sp.Device(0)  # use GPU 0; use sp.Device(-1) for CPU
    h5_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.h5')))

    for idx, file_path in enumerate(h5_files, start=1):
        # label outputs by index instead of original basename
        print(f"====== Processing {file_path} ======")
        output_label = f'{idx:03d}_calib'
        process_and_save_calibration(file_path, output_label, device)
        print(f'Saved calibration maps: {output_label}')
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

def patched_get_vars(self):
    ndim = len(self.img_shape)

    # --- 强制提取 mps_ker_width 为 float ---
    mps_ker_width = np.max(np.abs(self.mps))
    if hasattr(mps_ker_width, 'item'):
        mps_ker_width = mps_ker_width.item()
    else:
        mps_ker_width = float(mps_ker_width)
    self.mps_ker_width = mps_ker_width

    mps_ker_shape = [self.num_coils] + [self.mps_ker_width] * ndim
    if self.coord is None:
        img_ker_shape = [i + self.mps_ker_width - 1 for i in self.y.shape[1:]]
    else:
        if self.grd_shape is None:
            self.grd_shape = sp.estimate_shape(self.coord)
        img_ker_shape = [i + self.mps_ker_width - 1 for i in self.grd_shape]

    self.img_ker = sp.dirac(img_ker_shape, dtype=self.dtype, device=self.device)
    with self.device:
        self.mps_ker = self.device.xp.zeros(mps_ker_shape, dtype=self.dtype)


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

    # # Step 6: Jsense reconstruction
    # def step6():
    #     # Make sure inputs are clean numpy arrays (no mixed types)
    #     kspace_clean = np.array(cp.asnumpy(kspace_dev))
    #     calib_clean = np.array(cp.asnumpy(calib))

    #     # Compute mps_ker_width safely
    #     mps_ker_width = np.max(np.abs(calib_clean))
    #     if hasattr(mps_ker_width, 'item'):
    #         mps_ker_width = mps_ker_width.item()
    #     else:
    #         mps_ker_width = float(mps_ker_width)

    #     recon = mr.app.JsenseRecon(
    #         sp.to_device(kspace_clean, device=device),
    #         sp.to_device(calib_clean, device=device),
    #         lamda=0,
    #         device=device
    #     )
    #     recon.mps_ker_width = mps_ker_width  # manually override internal value
    #     result = recon.run()
    #     return result

    # recon_np = save_or_load('step6_recon.npy', step6)

    # Step 6: pure-NumPy JsenseRecon
    def step6_cpu():
        # 1) 准备纯 NumPy 输入
        ksp = np.array(kspace)       # kspace 已经是 numpy from step2
        cal = np.array(calib_np)     # calib_np 本来就是 numpy from step5

        # 2) 构建 JsenseRecon，用 CPU
        device = sp.Device(-1)
        recon = mr.app.JsenseRecon(
            ksp,
            cal,
            lamda=0,
            mps_ker_width=16,   # 或你想要的任何 Python int
            device=device
        )

        # 3) 直接跑（内部全用 numpy）
        img = recon.run()  # 得到一个 numpy array
        return img

    recon_np = save_or_load('step6_recon.npy', step6_cpu)

    # Final: show and save middle slice
    slice_idx = recon_np.shape[2] // 2
    plt.imshow(np.abs(recon_np[:, :, slice_idx]), cmap='gray')
    plt.title(f'Reconstructed Slice {slice_idx}')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'recon.png'))
    plt.close()
    print('Saved recon.png')

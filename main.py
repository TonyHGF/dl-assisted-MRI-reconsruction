import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

import utils
import bart_caller
import bart.python.cfl as cfl

def main():
    # ========== Dir Definition ==========
    output_dir = r"./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    kspace_main_path = r"./output/kspace_main"
    kspace_ref_path = r"./output/kspace_ref"

    # ========== Data Preparation ==========
    fileh5 = h5py.File("./ksp_data/brain_7t_kspace.h5", 'r')
    dataset = fileh5['dataset']
    raw_data = dataset['data']

    kspace_main, kspace_ref = None, None

    if not utils.file_exists(output_dir, kspace_main_path):
        ksp_data = [acq for acq in raw_data if acq['head']['number_of_samples'] == 768]
        kspace_main = utils.build_kspace(ksp_data)
        print(f"Build k-space main: {kspace_main.shape}")
        cfl.writecfl(kspace_main_path, kspace_main)
    else:
        kspace_main = cfl.readcfl(kspace_main_path)

    if not utils.file_exists(output_dir, kspace_ref_path):
        refscan_data = [acq for acq in raw_data if acq['head']['number_of_samples'] == 128]
        kspace_ref  = utils.build_kspace(refscan_data)
        print(f"Build k-space refscan: {kspace_ref.shape}")
        cfl.writecfl(kspace_ref_path, kspace_ref)
    else:
        kspace_ref = cfl.readcfl(kspace_ref_path)
    
    bart_caller.show_shape(kspace_main_path)
    bart_caller.show_shape(kspace_ref_path)

    # ========== 1D IFFT ==========



if __name__ == '__main__':
    main()
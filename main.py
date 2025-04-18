import h5py
import numpy as np
import matplotlib.pyplot as plt

import utils

def main():
    # ========== Data Preparation ==========
    fileh5 = h5py.File("./ksp_data/brain_7t_kspace.h5", 'r')
    dataset = fileh5['dataset']
    raw_data = dataset['data']
    ksp_data = [acq for acq in raw_data if acq['head']['number_of_samples'] == 768]
    refscan_data = [acq for acq in raw_data if acq['head']['number_of_samples'] == 128]

    kspace_main = utils.build_kspace(ksp_data)
    kspace_ref  = utils.build_kspace(refscan_data)
    print(f"Build Kspace:\n Main: {kspace_main.shape}\n RefScan: {kspace_ref.shape}")


if __name__ == '__main__':
    main()
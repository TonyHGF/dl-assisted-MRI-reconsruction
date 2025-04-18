import numpy as np


def summarize_dtype(dtype, indent=0):
    prefix = "  " * indent
    if dtype.fields is None:
        print(f"{prefix}{dtype}")
        return
    for name, (subdtype, offset) in dtype.fields.items():
        print(f"{prefix}{name}: ", end="")
        if subdtype.fields:
            print()  # newline
            summarize_dtype(subdtype, indent + 1)
        else:
            print(subdtype)


def build_kspace(acquisition_list):
    kx = acquisition_list[0]['head']['number_of_samples']
    channels = acquisition_list[0]['head']['active_channels']

    max_ky, max_kz = 0, 0
    for acq in acquisition_list:
        idx = acq['head']['idx']
        max_ky = max(max_ky, idx['kspace_encode_step_1'])
        max_kz = max(max_kz, idx['kspace_encode_step_2'])

    ky_dim = max_ky + 1
    kz_dim = max_kz + 1

    kspace_4d = np.zeros((kx, ky_dim, kz_dim, channels), dtype=np.complex64)

    for acq in acquisition_list:
        head = acq['head']
        idx = head['idx']
        data = acq['data'].view(np.complex64).reshape(channels, kx).T

        ky = idx['kspace_encode_step_1']
        kz = idx['kspace_encode_step_2']
        kspace_4d[:, ky, kz, :] = data

    return kspace_4d
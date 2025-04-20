import numpy as np
from pathlib import Path


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


def file_exists(folder_path: str, filename: str) -> bool:
    """
    Check if a file with the specified name (ignoring extensions) exists in the given folder.

    Parameters:
    folder_path (str): The folder where the search should be performed.
    filename (str): The target filename, which can be a full path or just a filename.

    Returns:
    bool: True if a file with the same name (ignoring extension) exists, otherwise False.
    """
    folder = Path(folder_path).resolve()  # Convert to absolute path
    file_name = Path(filename).stem  # Extract filename without extension

    # Ensure the folder exists
    if not folder.is_dir():
        raise ValueError(f"Folder '{folder}' does not exist or is not a valid directory.")

    # Check for matching filenames (ignoring extensions)
    return any(file.stem == file_name for file in folder.iterdir() if file.is_file())
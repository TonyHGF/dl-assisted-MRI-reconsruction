"""
A toolbox to call BART commands in python.
Author: Gangfeng Hu
"""


import os
import subprocess



def ifft(input_path, output_path, bitmask=1):
    """
    Runs the BART inverse FFT (ifft) command.

    Parameters:
        input_path (str): Path to the input data.
        output_path (str): Path to the output data.
        bitmask (int, optional): Bitmask specifying along which dimensions the FFT is performed. Default is 1.

    Usage:
        ifft("input", "output", bitmask=1)

    Equivalent to:
        bart fft -iu bitmask input output
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    command = f"bart fft -iu {bitmask} {input_path} {output_path}"
    subprocess.run(command, shell=True, cwd=script_dir, check=True)


def extract(input_path, output_path, *args):
    """
    Runs the BART extract command.

    Parameters:
        input_path (str): Path to the input data.
        output_path (str): Path to the output data.
        *args (tuple): Sequence of dimension, start, and end indices.
                       Should be in the form (dim1, start1, end1, dim2, start2, end2, ...).

    Usage:
        extract("input", "output", 0, 512, 513)

    Equivalent to:
        bart extract 0 512 513 input output
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(args) % 3 != 0:
        raise ValueError("Arguments must be in groups of three: (dim, start, end)")

    args_str = " ".join(map(str, args))
    command = f"bart extract {args_str} {input_path} {output_path}"

    subprocess.run(command, shell=True, cwd=script_dir, check=True)


def bitmask(*dims, reverse=False):
    """
    Converts between a bitmask and a set of dimensions.

    Parameters:
        *dims (int): Dimensions to be included in the bitmask.
        reverse (bool, optional): If True, converts a bitmask back to dimensions. Default is False.

    Returns:
        int or list: Bitmask value if converting dimensions to bitmask, 
                     or a list of dimensions if converting from a bitmask.

    Usage:
        bitmask(1, 2)        -> 6  (Equivalent to: bart bitmask 1 2)
        bitmask(6, reverse=True)  -> [1, 2]  (Equivalent to: bart bitmask -b 6)
    """
    if reverse:
        command = ["bart", "bitmask", "-b", str(dims[0])]
    else:
        command = ["bart", "bitmask"] + [str(dim) for dim in dims]

    result = subprocess.run(command, capture_output=True, text=True, check=True)

    output = result.stdout.strip()
    if reverse:
        return list(map(int, output.split())) if output else []
    return int(output)


def rss(input_path, output_path, bitmask):
    """
    Computes the root sum of squares (RSS) along selected dimensions.

    Parameters:
        input_path (str): Path to the input data.
        output_path (str): Path to the output data.
        bitmask (int): Bitmask specifying along which dimensions to compute RSS.

    Usage:
        rss("input", "output", 6)

    Equivalent to:
        bart rss 6 input output
    """
    command = f"bart rss {bitmask} {input_path} {output_path}"
    subprocess.run(command, shell=True, check=True)


def whiten(input_path, ndata_path, output_path, optmat_out=None, covar_out=None, 
                optmat_in=None, covar_in=None, normalize=False):
    """
    Applies multi-channel noise pre-whitening on input data using noise data.

    Parameters:
        input_path (str): Path to the input data.
        ndata_path (str): Path to the noise data.
        output_path (str): Path to save the whitened output.
        optmat_out (str, optional): Path to save the whitening matrix (optional).
        covar_out (str, optional): Path to save the noise covariance matrix (optional).
        optmat_in (str, optional): Path to an external whitening matrix.
        covar_in (str, optional): Path to an external noise covariance matrix.
        normalize (bool, optional): If True, normalizes variance to 1.

    Usage:
        whiten("input", "noise", "output", optmat_out="optmat_out", covar_out="covar_out")

    Equivalent to:
        bart whiten input noise output optmat_out covar_out
    """
    command = ["bart", "whiten"]

    if optmat_in:
        command.extend(["-o", optmat_in])
    if covar_in:
        command.extend(["-c", covar_in])
    if normalize:
        command.append("-n")

    command.extend([input_path, ndata_path, output_path])

    if optmat_out:
        command.append(optmat_out)
    if covar_out:
        command.append(covar_out)

    subprocess.run(command, check=True)


def coil_compression(kspace, output, num_virtual_channels=None, output_matrix=False, calibration_region=None, 
                          all_data=False, svd=False, geometric=False, espirit=False):
    """
    Performs coil compression on k-space data.

    Parameters:
        kspace (str): Path to the k-space input data.
        output (str): Path to save the compression coefficients or projected k-space.
        num_virtual_channels (int, optional): Number of virtual channels for compression.
        output_matrix (bool, optional): Output the compression matrix.
        calibration_region (int, optional): Size of the calibration region.
        all_data (bool, optional): Use all data to compute coefficients.
        svd (bool, optional): Use SVD-based compression.
        geometric (bool, optional): Use geometric compression.
        espirit (bool, optional): Use ESPIRiT-based compression.

    Usage:
        coil_compression("kspace", "coeff", num_virtual_channels=8, output_matrix=True)

    Equivalent to:
        bart cc -p 8 -M kspace coeff
    """
    command = ["bart", "cc"]

    if num_virtual_channels is not None:
        command.extend(["-p", str(num_virtual_channels)])
    if output_matrix:
        command.append("-M")
    if calibration_region is not None:
        calibration_region = str(calibration_region)
        command.extend(["-r", calibration_region])
    if all_data:
        command.append("-A")
    if svd:
        command.append("-S")
    if geometric:
        command.append("-G")
    if espirit:
        command.append("-E")

    command.extend([kspace, output])

    subprocess.run(command, check=True)


def apply_coil_compression(kspace, compression_matrix, projected_kspace, num_virtual_channels=None, 
                                inverse=False, no_fft_readout=False, svd=False, 
                                geometric=False, espirit=False):
    """
    Applies coil compression or its inverse operation.

    Parameters:
        kspace (str): Path to the k-space input data.
        compression_matrix (str): Path to the compression matrix.
        projected_kspace (str): Path to save the projected k-space data.
        num_virtual_channels (int, optional): Number of virtual channels for compression.
        inverse (bool, optional): Apply inverse operation.
        no_fft_readout (bool, optional): Do not apply FFT in the readout direction.
        svd (bool, optional): Use SVD-based compression.
        geometric (bool, optional): Use geometric compression.
        espirit (bool, optional): Use ESPIRiT-based compression.

    Usage:
        apply_coil_compression("kspace", "cc_matrix", "proj_kspace", num_virtual_channels=8)

    Equivalent to:
        bart ccapply -p 8 kspace cc_matrix proj_kspace
    """
    command = ["bart", "ccapply"]

    if num_virtual_channels is not None:
        command.extend(["-p", str(num_virtual_channels)])
    if inverse:
        command.append("-u")
    if no_fft_readout:
        command.append("-t")
    if svd:
        command.append("-S")
    if geometric:
        command.append("-G")
    if espirit:
        command.append("-E")

    command.extend([kspace, compression_matrix, projected_kspace])

    subprocess.run(command, check=True)

def get_dimensions(input_path):
    """
    Retrieves the size of the first four dimensions of the input data.

    Parameters:
        input_path (str): Path to the input data.

    Returns:
        tuple: A tuple containing the sizes of dimensions 0, 1, 2, and 3.

    Usage:
        sizes = get_dimensions("input")
    
    Equivalent to:
        bart -d 0 input
        bart -d 1 input
        bart -d 2 input
        bart -d 3 input
    """

    # Currently, the function is quite slow for unknown reason. 
    # Directly use the shape of k-space may be the better choice.
    # TODO: make it faster

    dimensions = []
    
    for dim in range(4):
        print("dim", dim)
        command = ["bart", "-d", str(dim), input_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        dimensions.append(int(result.stdout.strip()))

    return tuple(dimensions)

def ecalib(kspace_path, sensitivity_path, eigenvalue_map_path=None,
                kernel_size=None, calibration_size=None, num_maps=1,
                threshold=None, crop_value=None, soft_sense=False,
                soft_weighting=False, intensity_correction=False,
                first_part_only=False, no_phase_rotation=False,
                variance=None, auto_threshold=False, debug_level=None):
    """
    Runs BART ecalib to estimate coil sensitivity maps from k-space data.

    Parameters:
        kspace_path (str): Path to input k-space data (no .cfl suffix).
        sensitivity_path (str): Path to output sensitivity maps.
        eigenvalue_map_path (str, optional): Path to output eigenvalue maps.
        kernel_size (str, optional): Kernel size, e.g., "6:6".
        calibration_size (str, optional): Calibration region size, e.g., "48:48".
        num_maps (int, optional): Number of ESPIRiT maps to compute (default: 1).
        threshold (float, optional): Null-space threshold.
        crop_value (float, optional): Crop value.
        soft_sense (bool): Use Soft-SENSE.
        soft_weighting (bool): Apply soft-weighting to singular vectors.
        intensity_correction (bool): Enable intensity correction.
        first_part_only (bool): Only perform first part of calibration.
        no_phase_rotation (bool): Do not rotate phase.
        variance (float, optional): Noise variance in data.
        auto_threshold (bool): Automatically select threshold.
        debug_level (int, optional): Verbosity/debug level.

    Usage:
        ecalib("refscan", "maps", "evmaps", kernel_size="6:6", calibration_size="48:48")
    """
    command = ["bart", "ecalib"]

    if kernel_size is not None:
        command.extend(["-k", kernel_size])
    if calibration_size is not None:
        command.extend(["-r", calibration_size])
    if num_maps is not None:
        command.extend(["-m", str(num_maps)])
    if threshold is not None:
        command.extend(["-t", str(threshold)])
    if crop_value is not None:
        command.extend(["-c", str(crop_value)])
    if variance is not None:
        command.extend(["-v", str(variance)])
    if debug_level is not None:
        command.extend(["-d", str(debug_level)])

    if soft_sense:
        command.append("-S")
    if soft_weighting:
        command.append("-W")
    if intensity_correction:
        command.append("-I")
    if first_part_only:
        command.append("-1")
    if no_phase_rotation:
        command.append("-P")
    if auto_threshold:
        command.append("-a")

    command.extend([kspace_path, sensitivity_path])
    if eigenvalue_map_path:
        command.append(eigenvalue_map_path)

    # print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def resize(input_path, output_path, resize_dims, center=True, front=False):
    """
    Resizes a BART array along specified dimensions with zero-padding or truncation.

    Parameters:
        input_path (str): Path to the input file (no .cfl suffix).
        output_path (str): Path to the output file (no .cfl suffix).
        resize_dims (list of tuple): List of (dim_index, new_size) to resize.
        center (bool): If True, apply centered resizing (-c).
        front (bool): If True, apply front-padded resizing (-f).

    Usage:
        resize("infile", "outfile", [(1, 1022), (2, 119)], center=True)
    """
    command = ["bart", "resize"]

    if center:
        command.append("-c")
    if front:
        command.append("-f")

    for dim, size in resize_dims:
        command.extend([str(dim), str(size)])

    command.extend([input_path, output_path])

    # print("Running:", " ".join(command))
    subprocess.run(command, check=True)

def pics(kspace_path, sensitivity_path, output_path,
              regularization="Q:0.001", scaling=25.788,
              rescale=True, debug_level=0):
    """
    Run BART PICS (Parallel Imaging Compressed Sensing) reconstruction.

    Parameters:
        kspace_path (str): Path to k-space data.
        sensitivity_path (str): Path to coil sensitivity maps.
        output_path (str): Path to save the reconstructed image.
        regularization (str): Regularization type and value (e.g. 'Q:0.001').
        scaling (float): Inverse scaling factor for the data.
        rescale (bool): Whether to rescale image after reconstruction.
        debug_level (int): Debug verbosity level.
    """
    command = ["bart", "pics", f"-d{debug_level}", "-R", regularization, "-w", str(scaling)]
    if rescale:
        command.append("-S")
    command += [kspace_path, sensitivity_path, output_path]

    # print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def reshape_4d(input_path, output_path, new_shape=(1, 1022, 119, 24)):
    """
    Reshape the input to the specified first 4 dimensions using BART.

    Parameters:
        input_path (str): Path to the input .cfl (no suffix).
        output_path (str): Path to the output .cfl (no suffix).
        new_shape (tuple of 4 ints): The new shape for the first 4 dimensions.

    Equivalent to:
        bart reshape 15 1 1022 119 24 input output
    """
    assert len(new_shape) == 4, "Expected exactly 4 dimensions for reshape"
    reshape_bitmask = 15  # Binary 1111: modifying dim0, dim1, dim2, dim3
    shape_strs = list(map(str, new_shape))

    command = ["bart", "reshape", str(reshape_bitmask)] + shape_strs + [input_path, output_path]
    # print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def show_shape(file_path):
    """
    Prints the shape of a BART .cfl file using `bart show -m`.

    Parameters:
        file_path (str): Path prefix of the BART file (without .cfl/.hdr)

    Prints:
        - Dimensions if file exists
        - Error message if file is missing or command fails
    """
    cfl_path = file_path + ".cfl"
    hdr_path = file_path + ".hdr"

    if not (os.path.isfile(cfl_path) and os.path.isfile(hdr_path)):
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        result = subprocess.run(
            ["bart", "show", "-m", file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        print(f"Shape of {file_path}:", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error running BART show: {e.stderr.strip()}")
    


if __name__ == '__main__':
    ksp_raw = r"/mikBIG/shreya/bart-demo-ismrm-2024/data_7t_invivo/ksp_raw"
    refscan = r"/mikBIG/shreya/bart-demo-ismrm-2024/data_7t_invivo/refscan"
    output_dir = r"/home/tony_hu/output/20250301"


    # =================== Do IFFT along x-axis ===================

    ksp_x_kykz = os.path.join(output_dir, r"ksp_x_kykz")
    if not file_exists(output_dir, ksp_x_kykz):
        print(f"Call BART ifft, save at {ksp_x_kykz}")
        ifft(ksp_raw, ksp_x_kykz)

    refscan_x_kykz = os.path.join(output_dir, r"refscan_x_kykz")
    if not file_exists(output_dir, refscan_x_kykz):
        print(f"Call BART ifft for {refscan}, save at {refscan_x_kykz}")
        ifft(refscan, refscan_x_kykz)

    # ======================== 2D Slice ==========================

    ksp_slice = os.path.join(output_dir, r"ksp_slice")
    if not file_exists(output_dir, ksp_slice):
        print(f"Call BART extract, save at {ksp_slice}")
        extract(ksp_x_kykz, ksp_slice, 0, 512, 513)

    refscan_slice = os.path.join(output_dir, r"refscan_slice")
    if not file_exists(output_dir, refscan_slice):
        print(f"Call BART extract, save at {refscan_slice}")
        extract(refscan_x_kykz, refscan_slice, 0, 512, 513)

    im_slice = os.path.join(output_dir, r"im_slice")
    bitmask = bitmask(1, 2)
    if not file_exists(output_dir, im_slice):
        print(f"Call BART ifft, save at {im_slice}")
        ifft(ksp_slice, im_slice, bitmask)

    rss_im_slice = os.path.join(output_dir, r"rss_im_slice")
    if not file_exists(output_dir, rss_im_slice):
        print(f"Call BART rss, save at {rss_im_slice}")
        bitmask2 = bitmask(3)
        rss(im_slice, rss_im_slice, bitmask2)

    # ======================== Noise Prewhitening ==========================

    noise = r"/mikBIG/shreya/bart-demo-ismrm-2024/data_7t_invivo/noise"
    ksp_slice_white = os.path.join(output_dir, r"ksp_slice_white")
    if not file_exists(output_dir, ksp_slice_white):
        print(f"Call BART whiten for {ksp_slice}, save at {ksp_slice_white}")
        whiten(ksp_slice, noise, ksp_slice_white, optmat_out="optmat", covar_out="covar")
    
    refscan_slice_white = os.path.join(output_dir, r"refscan_slice_white")
    if not file_exists(output_dir, refscan_slice_white):
        print(f"Call BART whiten for {refscan_slice}, save at {refscan_slice_white}")
        whiten(refscan_slice, noise, refscan_slice_white, optmat_out="optmat", covar_out="covar")

    refscan_3d_white = os.path.join(output_dir, r"refscan_3d_white")
    if not file_exists(output_dir, refscan_3d_white):
        print(f"Call BART whiten for {refscan}, save at {refscan_3d_white}")
        whiten(refscan, noise, refscan_3d_white, optmat_out="optmat", covar_out="covar")

    ksp_3d_white = os.path.join(output_dir, r"ksp_3d_white")
    if not file_exists(output_dir, ksp_3d_white):
        print(f"Call BART whiten for {ksp_raw}, save at {ksp_3d_white}")
        whiten(ksp_raw, noise, ksp_3d_white, optmat_out="optmat", covar_out="covar")

    # ======================== Coil Compression ==========================

    cc_mat_svd = os.path.join(output_dir, r"cc_mat_svd")
    if not file_exists(output_dir, cc_mat_svd):
        print(f"Call BART Coil Compression Matrix of {refscan_3d_white}, save at {cc_mat_svd}")
        coil_compression(refscan_3d_white, cc_mat_svd, svd=True, num_virtual_channels=24, output_matrix=True, calibration_region=48)

    refscan_3d_white_cc = os.path.join(output_dir, r"refscan_3d_white_cc")
    if not file_exists(output_dir, refscan_3d_white_cc):
        print(f"Call BART Coil Compression for {refscan_3d_white}, save at {refscan_3d_white_cc}")
        apply_coil_compression(refscan_3d_white, cc_mat_svd, refscan_3d_white_cc, svd=True, num_virtual_channels=24)

    ksp_3d_white_cc = os.path.join(output_dir, r"ksp_3d_white_cc")
    if not file_exists(output_dir, ksp_3d_white_cc):
        print(f"Call BART Coil Compression for {ksp_3d_white}, save at {ksp_3d_white_cc}")
        apply_coil_compression(ksp_3d_white, cc_mat_svd, ksp_3d_white_cc, svd=True, num_virtual_channels=24)
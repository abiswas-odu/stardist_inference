import os
import numpy as np
import h5py
try:
    import pyklb
except ImportError:
    print("pyklb install missing! All klb format operations will fail. ")
import tifffile as tif
from roi_convertor.gen_rois import gen_roi_narray


from csbdeep.io import save_tiff_imagej_compatible

def read_image(image_path_file):
    """Read an image file in in klb/h5/tif/npy format.
    Args:
        image_path_file: Path to the image file in klb/h5/tif/npy format with the same extensions respectively.
    Returns:
        N-dimensional numpy array with the image pixels in 8-bit format
    Raises:
        ValueError: if the path is not found.
        ValueError: if the image is not in one of the required formats.
    """
    if not os.path.exists(image_path_file):
        raise ValueError(f"Image file not found '{image_path_file}'")
    elif not os.path.isfile(image_path_file):
        raise ValueError(f"Image not a file '{image_path_file}'")
    elif not (os.path.splitext(image_path_file)[1] == '.klb' or
              os.path.splitext(image_path_file)[1] == '.h5' or
              os.path.splitext(image_path_file)[1] == '.tif' or
              os.path.splitext(image_path_file)[1] == '.npy'):
        raise ValueError(f"Image file not in supported format with appropriate extension '{image_path_file}'")

    if image_path_file[-3:] == 'npy':
        Xi = np.load(image_path_file)
    elif image_path_file[-3:] == 'tif':
        Xi = tif.imread(image_path_file)
    elif (image_path_file[-2:] == 'h5'):
        him = h5py.File(image_path_file, 'r')
        Xi = him.get('Data')[:]
    elif (image_path_file[-3:] == 'klb'):
        Xi = pyklb.readfull(image_path_file)

    # convert to 8-bit by 4-bit shift
    Xi = Xi >> 4
    Xi = Xi.astype(dtype=np.uint8)
    print('loaded image shape:', Xi.shape)
    return Xi

def crop_image(Xi, row_1, row_2, col_1, col_2):
    """Crop the X/Y dimension of the N-dimensional numpy array representing the image.
    Args:
        Xi: N-dimensional numpy array with the image pixels.
        row_1: starting row
        row_2: ending row
        col_1: starting column
        col_2: ending column
    Returns:
        N-dimensional numpy array with the cropped image pixels
    """
    return Xi[:, row_1:row_2, col_1:col_2]

def crop_frames(Xi, frame_1, frame_2):
    """Crop the X/Y dimension of the N-dimensional numpy array representing the image.
    Args:
        Xi: N-dimensional numpy array with the image pixels.
        frame_1: starting frame
        frame_2: ending frame
    Returns:
        N-dimensional numpy array with the cropped frames removed
    """
    return Xi[frame_1:frame_2, :, :]

def write_image(labels, out_image_file, output_format, gen_roi):
    """Writes a N-dimensional numpy array in tif format
    Args:
        labels:  N-dimensional numpy array
        out_image_file: Path to the output image file including name.
        output_format: The segmentation output format klb/h5/tif/npy.
        gen_roi: Generate ROI if true
    Raises:
        ValueError: if the path is invalid.
    """

    segmentation_file_name = ""
    if output_format.upper() == "KLB":
        segmentation_file_name = out_image_file + ".klb"
        pyklb.writefull(labels, segmentation_file_name)
    elif output_format.upper() == "H5":
        segmentation_file_name = out_image_file + ".h5"
        hf = h5py.File(segmentation_file_name, 'w')
        hf.create_dataset('Data', data=labels)
        hf.close()
    elif output_format.upper() == "NPY":
        segmentation_file_name = out_image_file + ".npy"
        np.save(segmentation_file_name,labels)
    else:
        segmentation_file_name = out_image_file + ".tif"
        save_tiff_imagej_compatible(segmentation_file_name, labels.astype('uint16'), axes='ZYX')

    if gen_roi:
        gen_roi_narray(labels, segmentation_file_name)
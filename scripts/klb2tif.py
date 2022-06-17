import pyklb
import sys
import numpy as np
from csbdeep.io import save_tiff_imagej_compatible
import os

# Either file name or directory
klb_input = sys.argv[1]

if os.path.isdir(klb_input):
    # Walk the dir and get all klb files and convert them
    result = [os.path.join(dp, f)
              for dp, dn, filenames in os.walk(klb_input)
              for f in filenames if (os.path.splitext(f)[1] == '.klb')]
    for image_file in result:
        Xi = pyklb.readfull(image_file)
        out_base_name = os.path.splitext(os.path.basename(image_file))[0]
        op_file_name = out_base_name + ".tif"
        out_dir = os.path.dirname(image_file)
        op_file_name = os.path.join(out_dir,op_file_name)
        save_tiff_imagej_compatible(op_file_name, Xi.astype('uint16'), axes='ZYX')
elif os.path.isfile(klb_input):
    Xi = pyklb.readfull(klb_input)
    out_base_name = os.path.splitext(os.path.basename(klb_input))[0]
    op_file_name = out_base_name + ".tif"
    out_dir = os.path.dirname(klb_input)
    op_file_name = os.path.join(out_dir,op_file_name)
    save_tiff_imagej_compatible(op_file_name, Xi.astype('uint16'), axes='ZYX')
else:
    print("invalid file or directory.")


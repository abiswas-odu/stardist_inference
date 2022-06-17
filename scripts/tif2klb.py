import pyklb
import sys
import numpy as np
import os
import tifffile as tif

tif_input = sys.argv[1]

if os.path.isdir(tif_input):
    # Walk the dir and get all tif files and convert them
    result = [os.path.join(dp, f)
              for dp, dn, filenames in os.walk(tif_input)
              for f in filenames if (os.path.splitext(f)[1] == '.tif')]
    for image_file in result:
        Xi = tif.imread(image_file)
        out_base_name = os.path.splitext(os.path.basename(image_file))[0]
        op_file_name = out_base_name + ".klb"
        out_dir = os.path.dirname(image_file)
        op_file_name = os.path.join(out_dir,op_file_name)
        pyklb.writefull(Xi, op_file_name)

elif os.path.isfile(tif_input):
    Xi = tif.imread(tif_input)
    out_base_name = os.path.splitext(os.path.basename(tif_input))[0]
    op_file_name = out_base_name + ".klb"
    out_dir = os.path.dirname(tif_input)
    op_file_name = os.path.join(out_dir,op_file_name)
    pyklb.writefull(Xi, op_file_name)
else:
    print("invalid file or directory.")



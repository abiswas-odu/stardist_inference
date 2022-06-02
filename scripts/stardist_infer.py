from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
import h5py
import pyklb
from stardist import random_label_cmap
from stardist.models import StarDist3D
from stardist import fill_label_holes
#from skimage.external import tifffile as tif
from stardist.matching import matching, matching_dataset
import cv2
import os
from stardist import gputools_available
import tensorflow as tf
import argparse

# run 3D stardist please
def run_3D_stardist(input_image,output_dir):
    image_name = os.path.basename(input_image)

    if (input_image[-3:] == 'npy'):
        Xi = np.load(input_image)
        im_name = image_name[:-4]
    elif (input_image[-3:] == 'tif') or ( (input_image[-3:] == 'iff')):
        Xi = tif.imread(input_image)
        if (input_image[-3:] == 'tif'):
            im_name = image_name[:-4]
        else:
            im_name = image_name[:-5]
    elif (input_image[-2:] == 'h5'):
        im_name = image_name[:-3]
        him = h5py.File(input_image, 'r')
        Xi = him.get('dataset_1')[:]
        #Xi = Xi.astype(int)

        # HERE IS WHERE CROPPING OF AN IMAGE OCCURS IF THE IMAGE IS EXCESSIVELY LARGE
        # Xi = Xi[:, 1069:1739, 619:1269]
    elif (input_image[-3:] == 'klb'):
        im_name = image_name[:-4]
        Xi = pyklb.readfull(input_image)

        # HERE IS WHERE CROPPING OF AN IMAGE OCCURS IF THE IMAGE IS EXCESSIVELY LARGE
        # Xi = Xi[:, 300:812, 900:1412]
        # Xi = Xi[:, 0:1024, 0:1024]

    # convert to 8-bit by 4-bit shift
    Xi = Xi >> 4
    Xi = Xi.astype(dtype=np.uint8)

    print('image shape ',Xi.shape)

    bSplitPredict = False
    if bSplitPredict:

        prob_mat, dist_mat = model.predict(normalize(Xi, 1,99.8, axis=axis_norm))

        # should be TIME in name of output
        #np.save(output_dir + '/prob_mat' + im_name + '.npy',prob_mat)
        #np.save(output_dir + '/dist_mat' + im_name + '.npy', dist_mat)

        #prob_mat = np.load(output_dir + '/prob_mat' + im_name + '.npy')
        #dist_mat = np.load(output_dir + '/dist_mat' + im_name + '.npy')

        # This is the post processing involving nms suppression
        prob_thresh = 0.5
        nms_thresh = 0.3
        labels, details = model._instances_from_prediction(img_shape=Xi.shape, \
                                                           prob=prob_mat, \
                                                           dist=dist_mat, \
                                                           points=None, \
                                                           prob_class=None, \
                                                           prob_thresh=prob_thresh, \
                                                           nms_thresh=nms_thresh)

    else:
        # just perform both predict and post processing in one step
        labels, details = model.predict_instances(normalize(Xi, 1,99.8, axis=axis_norm))

    save_tiff_imagej_compatible(output_dir + '/Stardist3D_' + im_name + '.tif', labels.astype('uint16'),
                                axes='ZYX')


def get_x(dirname):
    datax_ = glob(dirname + "/images/*.npy")
    assert len(datax_) == 1
    return np.load(datax_[0])

def get_y(dirname):
    datay_ = glob(dirname + "/masks/*.npy")
    assert len(datay_) == 1
    return np.load(datay_[0])

def save_stardist_predictions(dirname):
    dirname_stardist_pred = dirname + "/stardist_pred"
    if not os.path.exists(dirname_stardist_pred):
        os.mkdir(dirname_stardist_pred)
    Xi = get_x(dirname)
    imfldr = dirname.split("/")[-1]

    save_tiff_imagej_compatible(dirname_stardist_pred + '/x_' + imfldr + '.tif', Xi, axes='ZYX')

    #Yi = get_y(dirname)
    #save_tiff_imagej_compatible(dirname_stardist_pred + '/y_' + imfldr + '.tif', Yi, axes='ZYX')

    labels, details = model.predict_instances(normalize(Xi, 1,99.8, axis=axis_norm))

    # labels is the predicted 3D image
    # details is a dictionary that contains the details of the predicted 3D image
    # its keys are
    # 'points' -- centroids of the blobs
    # 'dist' -- distances from the centroid to the boundary for each of the points
    # 'prob' -- probability that the centroid is a nuclei
    # 'rays_vertices' -- are the unit vectors along the rays used to contruct the image,
    # contains contributions from the Fibonacci spiral together with the anisotropy parameter, see Eq 1 in 3D Stardist paper.
    # 'ray_faces' -- not sure what it is, but never used it.

    save_tiff_imagej_compatible(dirname_stardist_pred + '/Stardist3D_' + imfldr + '.tif', labels.astype('uint16'), axes='ZYX')


# this version takes raw image either KLB or hdf5 or npy
# makes the crop on the fly and stores the information
# or takes a given cropping

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str,  default='.',
                        help='the path of the original raw intensity images')
    parser.add_argument('-o', '--output_path', type=str,  default='.',
                        help='output path - if left blank uses input_path/labels')
    parser.add_argument('-d', '--data_descriptor', type=str,  default='test_',
                        help='part of mask name and output files')
    parser.add_argument('-s', '--start', type=int,  default=1,
                        help='start of sequence - integer')
    parser.add_argument('-e', '--end', type=int,  default=1,
                        help='end of sequence - integer')
    parser.add_argument('-t', '--step', type=int,  default=1,
                        help='end of sequence - integer')
    parser.add_argument('-f','--format',type=str, default='',
                        help='format string for image name, for example image_%05d.klb')


args = parser.parse_args()

output_dir = args.output_path
input_path = args.image_path
data_descr = args.data_descriptor
format_str = args.format
start_frame = args.start
end_frame = args.end
step = args.step

np.random.seed(6)
# Check whether GPU is working
print(tf.__version__)
print(tf.test.gpu_device_name()) # runs inference ok w/o GPU ?

print('loading model ')
model = StarDist3D(None, name='LB_stardist_Jan2022All_32x256x256_flips', basedir='model')
print(model._thresholds)

# Here prob is the threshold beyond which the possible candidates for NMS are considered
# nms is the NMS threshold, if there is overlap beyond this threshold the NMS suppresses the overlapping nuclei
# note that model._thresholds is a tuple, must use model._thresholds._replace(prob=0.1, nms=0.3) to change the values, a
# simple assignment will throw an error
# The results are very sensitive to prob threshold, and not very sensitive to nms threshold
# prob=0.1, nms=0.3 work the best for visual inspection

model._thresholds = model._thresholds._replace(prob=0.5, nms=0.3)
print(model._thresholds)


n_channel = 1
axis_norm = (0, 1, 2)  # normalize channels independently
base_image_name = os.path.basename(os.path.normpath(input_path))


# if  several jobs at once - some get confused
try:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
except:
    if os.path.exists(output_dir):
        pass
    else:
        print('not able to make path', output_dir)



#for iT in range(start_frame,end_frame+1,step):
#        image_name = format_str % (iT,iT)  # for Hayden's data
#
#        time_str = '%05d' % iT
#
#        image_ext = image_name[-3:]
#        if (image_ext == 'npy'):
#            full_image_name = os.path.join(input_path,'npy_inputs',image_name)
#            output_dir = os.path.join(output_dir,'labels')
#        else:
#            full_image_name = os.path.join(input_path, image_name)
#        print('full image name ',full_image_name)
#        print('output to ',output_dir)
full_image_name = '/projects/LIGHTSHEET/posfailab/ab50/data/test_data/TestSets/out/folder_Cam_Long_00257.lux/klbOut_Cam_Long_00257.lux.h5'
run_3D_stardist(full_image_name, output_dir)


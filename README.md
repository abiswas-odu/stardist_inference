# stardist_inference: Perform inference using a stardist 3D model 

## Running on della.princeton.edu

You do not need to install anything. 

Make a copy of the script ```scripts/runInferenceSD.cmd``` and change the paths:

1. IMAGE_PATH: This should point to a folder with images in klb/h5/tif/npy format and extensions. The images may be directly under the folder or in sub-directories underneath. Additionally, this variable can also be a direct path to an image file.

2. OUT_DIR: The directory where the segmented labeled TIF files are saved.

3. MODEL_DIR: Can be changed if needed. Should not be necessary for a while.

NOTE: If you just want to test, the paths are already setup for a test. Just move on to the next step! 

Submit the job on the scheduler:
```sbatch runInferenceSD.cmd```

Wait for the job to be scheduled and run. The status can be checked with the command:
```squeue -u <USERNAME>```

Output files should be produced in OUT_DIR with the same basename as the input file, but the extension will be changed to ```.label.tif```. Also, review the SLURM log created in the same folder as the cmd script.

## Installing on your own machine

1. Clone the repo: 

```git clone https://github.com/abiswas-odu/stardist_inference```

2. Install with pip:

```
cd stardist_inference
pip install .
```
NOTE: This assumes that pyklb installs and runs correctly on your machine. The binary they have pre-built did not work for me. So, I had to install it: 

1. Build the kbl C++ library:
```
git clone https://bitbucket.org/fernandoamat/keller-lab-block-filetype.git
cd keller-lab-block-filetype
mkdir build
cd build
cmake ..
make
```
2. Point your environment variable to it: 
```export LD_LIBRARY_PATH=/projects/LIGHTSHEET/posfailab/ab50/tools/keller-lab-block-filetype/build/src```
   
## Running stardist_inference

stardist_inference has a reasonable commandline interface with help. 

```stardist_inference --help```

### SLURM Script

A sample SLURM script is provided in ```scripts/runInferenceSD.cmd```. Change the parameters on top and use ```sbatch``` to submit.

### Sample Commandline

```stardist_inference --image_path ${IMAGE_PATH} --output_dir ${OUT_DIR} --model_dir ${MODEL_DIR} --prob_thresh ${PROB_THRESH} --nms_thresh ${NMS_THRESH} --output_format tif --gen_roi```
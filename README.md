# stardist_inference: Perform inference using a stardist 3D model 

## Running on della.princeton.edu

You do not need to install anything. 

### Step 1
Make a copy of the script ```/projects/LIGHTSHEET/posfailab/ab50/tools/stardist_inference/scripts/runInferenceSD.cmd``` and change the paths:

```
cp /projects/LIGHTSHEET/posfailab/ab50/tools/stardist_inference/scripts/runInferenceSD.cmd <YOUR_PATH>
cd <YOUR_PATH>
```

### Step 2
Edit the script in vi/vim/nano/emacs or copy the script into your local machine and edit it: 

1. IMAGE_PATH: This should point to a folder with images in klb/h5/tif/npy format and extensions. The images may be directly under the folder or in sub-directories underneath. Additionally, this variable can also be a direct path to an image file.

2. OUT_DIR: The directory where the segmented labeled TIF files are saved.

3. OUT_FORMAT: The output format: klb/h5/tif/npy. 

NOTE: If you just want to test, the paths are already setup for a test. Just move on to the next step!

Save the script in ```<YOUR_PATH>```. 

### Step 3
Submit the job on the scheduler:
```sbatch runInferenceSD.cmd```

### Step 4
Wait for the job to be scheduled and run. The status can be checked with the command:
```squeue -u <USERNAME>```

Output files should be produced in OUT_DIR with the same basename as the input file, but the extension will be changed to ```.label.<ext>```. Also, review the SLURM log created in the same folder as the cmd script.

### Step 5
Converting files between klb and tif formats:

1. Setup environment
```
module load anaconda3/2021.11
conda activate /projects/LIGHTSHEET/posfailab/ab50/tools/tf2-posfai
export LD_LIBRARY_PATH=/projects/LIGHTSHEET/posfailab/ab50/tools/keller-lab-block-filetype/build/src
```

2. TIF to KLB format
```
python /projects/LIGHTSHEET/posfailab/ab50/tools/klb2tif.py <YOUR_KLB_FILE>
```

3. KLB to TIF format
```
python /projects/LIGHTSHEET/posfailab/ab50/tools/tif2klb.py <YOUR_TIF_FILE>
```

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
#!/bin/bash

#SBATCH --job-name=sd_infer      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=10              # total number of tasks across all nodes
#SBATCH --mem=250G               # total memory per node
#SBATCH --time=1:00:00           # total run time limit (HH:MM:SS)
#SBATCH -A molbio


IMAGE_PATH="/projects/LIGHTSHEET/posfailab/ab50/tools/stardist_inference/test/data/batch"
OUT_DIR="/projects/LIGHTSHEET/posfailab/ab50/tools/stardist_inference/test/output/batch"
OUT_FORMAT="klb"

##===================================================================================================
##=====================CHANGES BELOW THIS LINE FOR ADVANCED USERS====================================
##===================================================================================================

MODEL_DIR="/projects/LIGHTSHEET/posfailab/ab50/data/models/LB_stardist_Feb2022_32x256x256_Final_newsplit_flips"
PROB_THRESH="0.6"
NMS_THRESH="0.3"

##===================================================================================================
##==============================NO CHANGES BELOW THIS LINE===========================================
##===================================================================================================

echo Running on host `hostname`
echo Starting Time is `date`
echo Directory is `pwd`
starttime=$(date +"%s")

module purge
module load anaconda3/2020.11
export LD_LIBRARY_PATH=/projects/LIGHTSHEET/posfailab/ab50/tools/keller-lab-block-filetype/build/src
conda activate /projects/LIGHTSHEET/posfailab/ab50/tools/tf2-posfai

stardist_inference --image_path ${IMAGE_PATH} \
  --output_dir ${OUT_DIR} \
  --model_dir ${MODEL_DIR} \
  --prob_thresh ${PROB_THRESH} \
  --nms_thresh ${NMS_THRESH} \
  --output_format ${OUT_FORMAT} \
  --gen_roi

roi_convert generate-analytics --orig_image ${IMAGE_PATH} \
  --segmentation_image ${OUT_DIR} \
  --output_file ${OUT_DIR}\hand_correction_guide.csv

echo Ending time is $(date)
endtime=$(date +"%s")
diff=$(($endtime - $starttime))
echo Elapsed time is $(($diff/60)) minutes and $(($diff%60)) seconds.


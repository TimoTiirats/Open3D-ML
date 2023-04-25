#!/bin/bash
#SBATCH -p gpu 
#SBATCH -c 4 
#SBATCH --gres=gpu:1 

cd ../..
python scripts/run_pipeline.py tf -c ml3d/configs/pointpillars_lightcode.yml \
--dataset_path /home/timo/Open3D/LightCode_Dataset --pipeline ObjectDetection
